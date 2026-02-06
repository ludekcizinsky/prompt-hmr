#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# PromptHMR checkpoints
DATA_ROOT="${PROMPTHMR_DATA_ROOT:-/scratch/izar/cizinsky/pretrained/prompthmr}"
mkdir -p "${DATA_ROOT}/pretrain"
gdown --folder -O "${DATA_ROOT}/pretrain/" https://drive.google.com/drive/folders/1EQ7arZz135T-WpxkS_K1R_hjZp3prh-y?usp=share_link
gdown --folder -O "${DATA_ROOT}/pretrain/" https://drive.google.com/drive/folders/18SywG7Fc_iTfVNaikjHAZmy-A9I85eKv?usp=sharing

# Dataset annoations (evaluation only)
gdown --folder -O "${DATA_ROOT}/" https://drive.google.com/drive/folders/1JKGXTDGaSpJ1Cp-_ikLMsw7MO7OyymIe?usp=share_link

# Thirdparty checkpoints
gdown --folder -O "${DATA_ROOT}/pretrain/" https://drive.google.com/drive/folders/1OKhTdL1QVFH3f4hbIEa7jLANx4azuPi1?usp=sharing
gdown --fuzzy -O "${DATA_ROOT}/pretrain/camcalib_sa_biased_l2.ckpt" https://drive.google.com/file/d/1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO/view?usp=sharing
gdown --fuzzy -O "${DATA_ROOT}/pretrain/droidcalib.pth" https://drive.google.com/file/d/14hgb59Jk2Pvfiqy4nntE7dUrcKgFmKSj/view?usp=sharing
gdown --fuzzy -O "${DATA_ROOT}/pretrain/vitpose-h-coco_25.pth" https://drive.google.com/file/d/1ZprPoNXe_f9a9flr0RhS3XCJBfqhFSeE/view?usp=sharing
wget -P "${DATA_ROOT}/pretrain/" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P "${DATA_ROOT}/pretrain/" https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth


# PHMR-Vid checkpoint (Google Drive file)
mkdir -p "${DATA_ROOT}/pretrain/phmr_vid"
gdown --fuzzy -O "${DATA_ROOT}/pretrain/phmr_vid/" https://drive.google.com/file/d/13yrY33o_iB27XCiekDkxogExAi2fG2rL/view?usp=drive_link

# Examples
gdown --folder -O "${DATA_ROOT}/" https://drive.google.com/drive/folders/1uhy_8rCjOELqR9G5BXBKu0-cnQOSHFkD?usp=sharing
