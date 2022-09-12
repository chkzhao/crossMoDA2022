#! /bin/bash

source /ocean/projects/asc170022p/findlamp/miniconda3/etc/profile.d/conda.sh
conda activate base

python img_resample.py
python sub_eval.py
python inv_img_resample.py
python post_label.py