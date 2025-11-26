#!/usr/bin/env python3
import os
import os.path as osp
import torch

# Defining the datasets we want and where logs live
model_types = ['infogcn2', 'msg3d', 'stgcn2']
datasets = {
    'ntu': ['CS', 'CV'],
    'ntu120': ['CSub', 'CSet'],
    'ucf101': ['1', '2', '3']}
# datasets = ['ntu', 'ntu120', 'ucf101']
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


def get_completion_from_logs():
    root = './logs'
    for model_type in model_types:
        print(model_type)
        for dataset in datasets:
            # Status_dict = {'dataset': {'evaluation': {'completed': ...}}}
            status_dict[dataset] = {}
            evals = os.listdir(osp.join(root, model_type, dataset))
            print(f"   {dataset}")
            for evaluation in evals:
                print(f"     {evaluation}")
                run_status = {'training': [],
                            'failed': [],
                            'timeout': [],
                            'completed': [],
                            'evaluated': []}
                # eg.                 ./logs/infogcn2/ntu/CS/train/
                train_folder_pth = osp.join(root, model_type, dataset, evaluation, 'train')
                eval_folder_pth = osp.join(root, model_type, dataset, evaluation, 'eval')
                train_log_files = [osp.join(train_folder_pth, log_file) for log_file in
                            os.listdir(train_folder_pth) if 'train' in log_file]
                try:
                    eval_log_files = [osp.join(eval_folder_pth, log_file) for log_file in
                                    os.listdir(eval_folder_pth) if 'eval' in log_file]
                except FileNotFoundError:
                    pass
                # Check status of log_file
                for log_file in train_log_files:
                    check_failed(log_file, run_status)
                # Check if that run has been evaluated
                for log_file in eval_log_files:
                    eval_runname = log_file.split('/')[-1].replace('_', ' ')[5:-4]
                    if eval_runname in run_status['completed']:
                        run_status['evaluated'].append(eval_runname)
                status_dict[dataset][evaluation] = run_status
                print(f"       {run_status}")


def get_checkpoint_results(path, model_type):
    checkpoint = torch.load(path, map_location='cpu')
    try:
        results = checkpoint['results']
        if model_type=='infogcn2':
            best_res = torch.tensor(results['test_ACC'][-1]).max().item()
        else:
            best_res = results['test_ACC'][-1].max().item()
        return best_res
    except KeyError as e:
        print(f"Key {e} does not exist...")
        return None
    except IndexError:
        print("Training likely didn't finish...")
        return None


def get_test_results():
    root="./results/"
    results = {}
    for model_type in model_types:
        for dataset, evals in datasets.items():
            for evaluation in evals:
                # Define the root path for the models
                model_res_root = osp.join(root, model_type, dataset, evaluation, 'train')
                res_names = [
                    path for path in os.listdir(model_res_root) if
                    path.startswith(f"{model_type}_{dataset}_{evaluation}")
                ]
                # Truly one of the worst ways to get just the 'base' and 'cnn' models...
                res_paths = {}
                for res_name in res_names:
                    if ('cnn' in res_name or 'base' in res_name):
                        res_paths[res_name] = osp.join(model_res_root, res_name)

                # Append the result name and accuracy to the results dict
                for res_name, res_path in res_paths.items():
                    results[res_name[:-3]] = get_checkpoint_results(res_path, model_type)
    return results



if __name__=="__main__":
    results = get_test_results()
    print(results)
