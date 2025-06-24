#!/bin/bash

module load cuda/12.8
module load miniforge3/24.11
source activate ML3.10

python start_train_gtn.py