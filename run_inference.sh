#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate phmr_pt2.4 
module load gcc ffmpeg

scene_dir_path=$1

repo_path=/home/cizinsky/master-thesis
cd $repo_path/submodules/prompthmr

python scripts/preprocess_scene.py --scene-dir $scene_dir_path
