#!/bin/bash
# use the bash shell

source /ocean/projects/asc170022p/findlamp/miniconda3/etc/profile.d/conda.sh
conda activate base

eval "$*" # The argument passed to this script should be the command to launch a python script