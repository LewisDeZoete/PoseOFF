#!/usr/bin/env python3
import os
import os.path as osp

# Defining the datasets we want and where logs live
datasets = ['ntu', 'ntu120', 'ucf101']
root = './logs'
status_dict = {} # We will use this to store the full status of all the runs

# Using these to check the status of runs
actions = {
    "(FAILED)": lambda fname, run_status: run_status['failed'].append(fname),
    "(TIMEOUT)": lambda fname, run_status: run_status['timeout'].append(fname),
    "(COMPLETED)": lambda fname, run_status: run_status['completed'].append(fname)
}

def check_failed(fname, run_status):
    with open(fname) as file:
        # Just getting the run name effectively
        fname = fname.split('/')[-1].replace('_', ' ')[6:-4]
        run_status['training'].append(fname) # by default, assume it's training...
        # check last 30 lines just to be safe?
        for line in file.readlines()[-25:]:
            for key, func in actions.items():
                # check if "failed", "timeout", "completed" in the line
                if key in line:
                    func(fname, run_status)
                    run_status['training'].remove(fname) # it's not training!
                    break
    # Sort the status lists...
    for key, status_list in run_status.items():
        status_list.sort()
        run_status[key] = status_list



for dataset in datasets:
    # Status_dict = {'dataset': {'evaluation': {'completed': ...}}}
    status_dict[dataset] = {}
    evals = os.listdir(osp.join(root, dataset))
    print(dataset)
    for evaluation in evals:
        print(f"  {evaluation}")
        run_status = {'training': [],
                      'failed': [],
                      'timeout': [],
                      'completed': [],
                      'evaluated': []}
        # eg.                 ./logs/ntu/CS/train/
        train_folder_pth = osp.join(root, dataset, evaluation, 'train')
        eval_folder_pth = osp.join(root, dataset, evaluation, 'eval')
        train_log_files = [osp.join(train_folder_pth, log_file) for log_file in
                     os.listdir(train_folder_pth) if 'train' in log_file]
        try:
            eval_log_files = [osp.join(eval_folder_pth, log_file) for log_file in
                            os.listdir(eval_folder_pth) if 'eval' in log_file]
        except FileNotFoundError:
            print(f"Eval folder does not yet exist for {dataset}")
        # Check status of log_file
        for log_file in train_log_files:
            check_failed(log_file, run_status)
        # Check if that run has been evaluated
        for log_file in eval_log_files:
            eval_runname = log_file.split('/')[-1].replace('_', ' ')[5:-4]
            if eval_runname in run_status['completed']:
                run_status['evaluated'].append(eval_runname)
        status_dict[dataset][evaluation] = run_status
        print(f"    {run_status}")
