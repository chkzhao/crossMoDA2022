#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate crossMoDA

#python img_resample.py
#python sub_eval.py
python inv_img_resample.py
python post_label.py