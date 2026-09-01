#!/usr/bin/env python3

import argparse
from data_gen.utils import get_mean_map

parser = argparse.ArgumentParser(description='NTU-RGB-D Data Preparation')
parser.add_argument(
    '-d',
    dest='dataset',
    default='ntu',
    help='Dataset, either `ntu` or `ntu120` (default=ntu)'
)
parser.add_argument(
    '-f',
    dest='flow_embedding',
    default="cnn",
    help="Flow embedding (base or cnn)"
)
parser.add_argument(
    '-e',
    dest='evaluation',
    help='Evaluation (CS/CV, CSub/CSet)'
)
parser.add_argument(
    '--data_path_overwrite',
    help="Data path overwrite"
)
parser.add_argument(
    '--save_path_overwrite',
    help='Save path overwrite'
)
args = parser.parse_args()
