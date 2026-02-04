#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate phmr_pt2.4 
module load gcc ffmpeg

examples_path=/scratch/izar/cizinsky/pretrained/prompthmr/examples

python scripts/demo_video.py --input_video $examples_path/boxing.mp4
