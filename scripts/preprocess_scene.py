import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from pipeline import Pipeline
from pipeline.tools import est_camera
from pipeline.spec import run_cam_calib


def _list_image_frames(image_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png")
    frame_paths = []
    for ext in exts:
        frame_paths.extend(Path(image_dir).glob(ext))

    frame_paths = sorted(frame_paths, key=lambda p: p.name)
    if not frame_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    frame_names = [p.stem for p in frame_paths]
    return frame_paths, frame_names


def _load_images(frame_paths):
    images = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        images.append(img[..., ::-1])  # BGR -> RGB
    return np.stack(images, axis=0)


def _init_results():
    return {
        'camera': {},
        'people': {},
        'timings': {},
        'masks': None,
        'has_tracks': False,
        'has_hps_cam': False,
        'has_hps_world': False,
        'has_slam': False,
        'has_hands': False,
        'has_2d_kpts': False,
        'has_post_opt': False,
    }


def _to_numpy(d):
    for k, v in d.items():
        if isinstance(v, dict):
            _to_numpy(v)
        elif hasattr(v, "detach"):
            d[k] = v.detach().cpu().numpy()


def _split_smplx_pose(pose_aa):
    # pose_aa: (P, 55*3)
    root_pose = pose_aa[:, 0:3]
    body_pose = pose_aa[:, 3:66].reshape(-1, 21, 3)
    jaw_pose = pose_aa[:, 66:69]
    leye_pose = pose_aa[:, 69:72]
    reye_pose = pose_aa[:, 72:75]
    lhand_pose = pose_aa[:, 75:120].reshape(-1, 15, 3)
    rhand_pose = pose_aa[:, 120:165].reshape(-1, 15, 3)
    return root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose


def main(scene_dir: str, static_camera: bool = False):
    scene_dir = os.path.abspath(scene_dir)
    images_root = os.path.join(scene_dir, "images")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Missing images directory: {images_root}")

    cam_ids = [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
    if len(cam_ids) != 1:
        raise ValueError(f"Expected exactly one cam_id under {images_root}, found: {cam_ids}")

    cam_id = cam_ids[0]
    image_dir = os.path.join(images_root, cam_id)

    frame_paths, frame_names = _list_image_frames(image_dir)
    images = _load_images(frame_paths)

    pipeline = Pipeline(static_cam=static_camera)
    pipeline.images = images
    pipeline.seq_folder = scene_dir
    pipeline.cfg.seq_folder = scene_dir
    pipeline.cfg.img_folder = image_dir
    pipeline.fps = 30
    pipeline.results = _init_results()

    if not pipeline.results['has_slam']:
        pipeline.results['camera'] = est_camera(images[0])

    if not pipeline.results['has_slam']:
        stride = len(images) // 30
        if stride == 0:
            stride = 1
        spec_calib = run_cam_calib(
            images,
            out_folder=None,
            save_res=False,
            stride=stride,
            method='spec',
            first_frame_idx=0,
        )
        pipeline.results['spec_calib'] = spec_calib

    if not pipeline.results['has_tracks']:
        print("Running detect, segment, and track pipeline...")
        pipeline.run_detect_track()

    if not pipeline.results['has_slam']:
        print("Running camera motion estimation...")
        pipeline.camera_motion_estimation(static_cam=static_camera)

    if not pipeline.results['has_2d_kpts']:
        print("Estimating 2D keypoints...")
        pipeline.estimate_2d_keypoints()

    if not pipeline.results['has_hps_cam']:
        print("Running human mesh estimation...")
        pipeline.hps_estimation()

    if not pipeline.results['has_hps_world']:
        print("Running world coordinates estimation...")
        pipeline.world_hps_estimation()

    _to_numpy(pipeline.results)

    if pipeline.cfg.run_post_opt and not pipeline.results['has_post_opt']:
        print("Running post optimization...")
        pipeline.post_optimization()
        _to_numpy(pipeline.results)

    results = pipeline.results
    num_frames = images.shape[0]

    # Output directories
    all_cameras_dir = os.path.join(scene_dir, "all_cameras", cam_id)
    seg_all_dir = os.path.join(scene_dir, "seg", "img_seg_mask", cam_id, "all")
    smplx_dir = os.path.join(scene_dir, "smplx")
    os.makedirs(all_cameras_dir, exist_ok=True)
    os.makedirs(seg_all_dir, exist_ok=True)
    os.makedirs(smplx_dir, exist_ok=True)

    # Prepare per-person mask dirs (zero-based contiguous IDs)
    track_ids = sorted(results['people'].keys())
    id_map = {tid: idx for idx, tid in enumerate(track_ids)}
    for _, pid in id_map.items():
        os.makedirs(os.path.join(scene_dir, "seg", "img_seg_mask", cam_id, str(pid)), exist_ok=True)

    # Union masks
    union_masks = results.get('masks', None)
    if union_masks is None:
        union_masks = np.zeros((num_frames, images.shape[1], images.shape[2]), dtype=bool)
        for tid in track_ids:
            person = results['people'][tid]
            for f_idx, m in zip(person['frames'], person['masks']):
                if f_idx < num_frames:
                    union_masks[f_idx] |= m

    # Build per-person frame->mask lookup
    person_masks = {}
    for tid in track_ids:
        person = results['people'][tid]
        frames = person['frames'].astype(int)
        masks = person['masks']
        person_masks[tid] = {int(f): masks[i] for i, f in enumerate(frames)}

    # Smplx per-person frame->idx lookup
    person_frames = {}
    for tid in track_ids:
        person = results['people'][tid]
        frames = person['frames'].astype(int)
        person_frames[tid] = {int(f): i for i, f in enumerate(frames)}

    # Camera intrinsics (same for all frames)
    cam_world = results['camera_world']
    img_focal = cam_world.get('img_focal', results['camera'].get('img_focal'))
    img_center = cam_world.get('img_center', results['camera'].get('img_center'))
    if isinstance(img_focal, (list, tuple, np.ndarray)):
        fx = float(img_focal[0])
        fy = float(img_focal[1]) if len(img_focal) > 1 else float(img_focal[0])
    else:
        fx = fy = float(img_focal)
    cx = float(img_center[0])
    cy = float(img_center[1])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Determine missing frames
    skip_frames = set()
    frame_ids = []
    if all(name.isdigit() for name in frame_names):
        frame_ids = [int(n) for n in frame_names]
        expected = set(range(min(frame_ids), max(frame_ids) + 1))
        missing_ids = expected - set(frame_ids)
        skip_frames.update(missing_ids)

    # Write outputs per frame
    for i, frame_name in enumerate(frame_names):
        # all_cameras
        Rcw = cam_world['Rcw'][i]
        Tcw = cam_world['Tcw'][i]
        # Save world->camera extrinsics to match visualization expectations.
        extr = np.concatenate([Rcw, Tcw.reshape(3, 1)], axis=1).astype(np.float32)
        np.savez(
            os.path.join(all_cameras_dir, f"{frame_name}.npz"),
            intrinsics=K[None, ...],
            extrinsics=extr[None, ...],
        )

        # seg all
        if union_masks is not None and i < union_masks.shape[0]:
            umask = union_masks[i]
        else:
            umask = np.zeros(images.shape[1:3], dtype=bool)
        umask_img = (umask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(seg_all_dir, f"{frame_name}.png"), umask_img)

        # per-person seg masks
        for tid in track_ids:
            pid = id_map[tid]
            m = person_masks[tid].get(i, None)
            if m is None:
                m = np.zeros(images.shape[1:3], dtype=bool)
            m_img = (m.astype(np.uint8) * 255)
            out_path = os.path.join(scene_dir, "seg", "img_seg_mask", cam_id, str(pid), f"{frame_name}.png")
            cv2.imwrite(out_path, m_img)

        # smplx
        frame_pose = []
        frame_shape = []
        frame_trans = []
        for tid in track_ids:
            idx = person_frames[tid].get(i, None)
            if idx is None:
                continue
            smplx_world = results['people'][tid]['smplx_world']
            frame_pose.append(smplx_world['pose'][idx])
            frame_shape.append(smplx_world['shape'][idx])
            frame_trans.append(smplx_world['trans'][idx])

        if len(frame_pose) == 0:
            skip_frames.add(int(frame_name) if frame_name.isdigit() else i)
            empty_root = np.zeros((0, 3), dtype=np.float32)
            empty_body = np.zeros((0, 21, 3), dtype=np.float32)
            empty_jaw = np.zeros((0, 3), dtype=np.float32)
            empty_eye = np.zeros((0, 3), dtype=np.float32)
            empty_hand = np.zeros((0, 15, 3), dtype=np.float32)
            empty_betas = np.zeros((0, 10), dtype=np.float32)
            empty_trans = np.zeros((0, 3), dtype=np.float32)
            np.savez(
                os.path.join(smplx_dir, f"{frame_name}.npz"),
                betas=empty_betas,
                root_pose=empty_root,
                body_pose=empty_body,
                jaw_pose=empty_jaw,
                leye_pose=empty_eye,
                reye_pose=empty_eye,
                lhand_pose=empty_hand,
                rhand_pose=empty_hand,
                trans=empty_trans,
            )
            continue

        pose_aa = np.stack(frame_pose, axis=0)
        shape = np.stack(frame_shape, axis=0)
        trans = np.stack(frame_trans, axis=0)
        root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose = _split_smplx_pose(pose_aa)

        np.savez(
            os.path.join(smplx_dir, f"{frame_name}.npz"),
            betas=shape[:, :10],
            root_pose=root_pose,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            lhand_pose=lhand_pose,
            rhand_pose=rhand_pose,
            trans=trans,
        )

    # Write skip_frames.csv if needed
    if skip_frames:
        skip_path = os.path.join(scene_dir, "skip_frames.csv")
        with open(skip_path, "w", encoding="utf-8") as f:
            f.write(", ".join(str(x) for x in sorted(skip_frames)))


if __name__ == "__main__":
    tyro.cli(main)
