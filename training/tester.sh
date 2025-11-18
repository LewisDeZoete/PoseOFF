#!/usr/bin/env sh

source ../environment/bin/activate

srun python training/train_infogcn.py
