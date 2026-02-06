import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from pipeline import Pipeline
from pipeline.tools import est_camera
from pipeline.spec import run_cam_calib
from data_config import BODY_MODELS_ROOT
from submodules.smplx import smplx

import torch


def _to_numpy_array(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_stage_smplx(results, out_dir: Path, stage_key: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for track_id, person in sorted(results.get("people", {}).items()):
        frames = _to_numpy_array(person.get("frames"))
        smplx_data = person.get(stage_key)
        if frames is None or smplx_data is None:
            continue

        save_dict = {"frames": frames.astype(np.int32)}
        for key in ("pose", "shape", "trans", "rotmat", "contact", "static_conf_logits"):
            if key in smplx_data and smplx_data[key] is not None:
                save_dict[key] = _to_numpy_array(smplx_data[key])

        np.savez(out_dir / f"track_{track_id}.npz", **save_dict)

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


def _load_smplx_layer(device: str):
    layer = smplx.create(
        str(Path(BODY_MODELS_ROOT)),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
    )
    return layer.to(device)


def _render_smplx_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(1.0, 1.0, 1.0))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        baseColorFactor=(1.0, 1.0, 1.0, 1.0),
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene.add(pr_mesh)

    fx, fy, cx, cy = (
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
    )
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
    w2c_cv = np.eye(4, dtype=np.float32)
    w2c_cv[:3, :4] = extrinsics.astype(np.float32)
    cv_to_gl = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    w2c_gl = cv_to_gl @ w2c_cv
    c2w_gl = np.linalg.inv(w2c_gl)
    scene.add(camera, pose=c2w_gl)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    mask = color_rgba[..., 3] > 0
    return mask


def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _save_silhouette_debug(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_path: Path,
    iou: float,
) -> None:
    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[gt_mask] = (0, 0, 255)  # blue
    overlay[pred_mask] = (255, 165, 0)  # orange (overlays gt)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(f"IoU: {iou:.3f}")
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pad_or_truncate(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(vec.shape[-1])
    if current_dim == target_dim:
        return vec
    if current_dim > target_dim:
        return vec[..., :target_dim]
    pad = torch.zeros((*vec.shape[:-1], target_dim - current_dim), device=vec.device, dtype=vec.dtype)
    return torch.cat([vec, pad], dim=-1)


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
    # Use PromptHMR mean-hand convention for hand pose defaults.
    pipeline.cfg.use_mean_hands = True
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
        _save_stage_smplx(
            pipeline.results,
            Path(scene_dir) / "misc" / "prompthmr" / "camera_smplx",
            "smplx_cam",
        )

    if not pipeline.results['has_hps_world']:
        print("Running world coordinates estimation...")
        pipeline.world_hps_estimation()
        _save_stage_smplx(
            pipeline.results,
            Path(scene_dir) / "misc" / "prompthmr" / "world_smplx",
            "smplx_world",
        )

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

        # per-person seg masks
        union_mask = np.zeros(images.shape[1:3], dtype=bool)
        for tid in track_ids:
            pid = id_map[tid]
            m = person_masks[tid].get(i, None)
            if m is None:
                m = np.zeros(images.shape[1:3], dtype=bool)
            union_mask |= m
            m_img = (m.astype(np.uint8) * 255)
            out_path = os.path.join(scene_dir, "seg", "img_seg_mask", cam_id, str(pid), f"{frame_name}.png")
            cv2.imwrite(out_path, m_img)

        # seg all (union of per-person masks)
        umask_img = (union_mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(seg_all_dir, f"{frame_name}.png"), umask_img)

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

    # Debug silhouettes: GT mask vs SMPL-X rendered mask.
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    smplx_layer = _load_smplx_layer(device)
    smplx_faces = np.asarray(smplx_layer.faces, dtype=np.int32)
    debug_root = Path(scene_dir) / "misc" / "prompthmr" / "silhouettes"
    img_h, img_w = images.shape[1:3]

    iou_sums = {pid: 0.0 for pid in range(len(track_ids))}
    iou_counts = {pid: 0 for pid in range(len(track_ids))}

    for i, frame_name in enumerate(frame_names):
        smplx_path = Path(smplx_dir) / f"{frame_name}.npz"
        if not smplx_path.exists():
            continue
        smplx_data = np.load(smplx_path)
        if "betas" not in smplx_data:
            continue

        betas = smplx_data["betas"]
        root_pose = smplx_data["root_pose"]
        body_pose = smplx_data["body_pose"]
        jaw_pose = smplx_data["jaw_pose"]
        leye_pose = smplx_data["leye_pose"]
        reye_pose = smplx_data["reye_pose"]
        lhand_pose = smplx_data["lhand_pose"]
        rhand_pose = smplx_data["rhand_pose"]
        trans = smplx_data["trans"]

        if betas.shape[0] == 0:
            continue

        # Camera params for this frame.
        cam_npz = np.load(os.path.join(all_cameras_dir, f"{frame_name}.npz"))
        intr = cam_npz["intrinsics"][0]
        extr = cam_npz["extrinsics"][0]

        for pid in range(betas.shape[0]):
            mask_path = Path(scene_dir) / "seg" / "img_seg_mask" / cam_id / str(pid) / f"{frame_name}.png"
            if not mask_path.exists():
                continue
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                continue
            gt_mask = gt_mask > 0

            with torch.no_grad():
                betas_tensor = torch.tensor(
                    betas[pid : pid + 1], dtype=torch.float32, device=device
                )
                expected_betas = int(getattr(smplx_layer, "num_betas", betas_tensor.shape[-1]))
                betas_tensor = _pad_or_truncate(betas_tensor, expected_betas)

                expr_tensor = None
                expected_expr = int(getattr(smplx_layer, "num_expression_coeffs", 0))
                if expected_expr > 0:
                    expr_tensor = torch.zeros(
                        (1, expected_expr), dtype=betas_tensor.dtype, device=betas_tensor.device
                    )

                def _maybe_flatten(x):
                    if x.ndim == 3:
                        return x.reshape(x.shape[0], -1)
                    return x

                output = smplx_layer(
                    betas=betas_tensor,
                    global_orient=_maybe_flatten(
                        torch.tensor(root_pose[pid : pid + 1], dtype=torch.float32, device=device)
                    ),
                    body_pose=_maybe_flatten(
                        torch.tensor(body_pose[pid : pid + 1], dtype=torch.float32, device=device)
                    ),
                    jaw_pose=_maybe_flatten(
                        torch.tensor(jaw_pose[pid : pid + 1], dtype=torch.float32, device=device)
                    ),
                    leye_pose=_maybe_flatten(
                        torch.tensor(leye_pose[pid : pid + 1], dtype=torch.float32, device=device)
                    ),
                    reye_pose=_maybe_flatten(
                        torch.tensor(reye_pose[pid : pid + 1], dtype=torch.float32, device=device)
                    ),
                    left_hand_pose=_maybe_flatten(
                        torch.tensor(lhand_pose[pid : pid + 1], dtype=torch.float32, device=device)
                    ),
                    right_hand_pose=_maybe_flatten(
                        torch.tensor(rhand_pose[pid : pid + 1], dtype=torch.float32, device=device)
                    ),
                    transl=torch.tensor(trans[pid : pid + 1], dtype=torch.float32, device=device),
                    expression=expr_tensor if expr_tensor is not None else None,
                )
            verts = output.vertices.detach().cpu().numpy()[0]
            pred_mask = _render_smplx_silhouette(
                verts, smplx_faces, intr, extr, img_w, img_h
            )
            iou = _compute_iou(gt_mask, pred_mask)
            iou_sums[pid] = iou_sums.get(pid, 0.0) + iou
            iou_counts[pid] = iou_counts.get(pid, 0) + 1
            out_path = debug_root / str(pid) / f"{frame_name}.png"
            _save_silhouette_debug(gt_mask, pred_mask, out_path, iou)

    results_path = debug_root / "results.txt"
    with results_path.open("w", encoding="utf-8") as f:
        for pid in sorted(iou_sums.keys()):
            count = iou_counts.get(pid, 0)
            mean_iou = (iou_sums[pid] / count) if count > 0 else 0.0
            f.write(f"person {pid}: {mean_iou:.6f}\n")


if __name__ == "__main__":
    tyro.cli(main)
