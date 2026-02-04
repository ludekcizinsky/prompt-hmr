#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate phmr_pt2.4 
module load gcc ffmpeg

repo_path=/home/cizinsky/master-thesis
cd $repo_path/submodules/prompthmr
scene_dir_path=/scratch/izar/cizinsky/thesis/v2_preprocessing/hi4d_pair15_fight
python scripts/preprocess_scene.py --scene-dir $scene_dir_path
