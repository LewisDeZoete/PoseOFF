import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

dataset = "ntu"
evaluation = "CS"
model_type = "cnn"

all_classes_path = './NTU120_classes_MACRO(Classes).csv'
classes_60_path = './NTU120_classes_MACRO(Class_categories_60).csv'
classes_120_path = './NTU120_classes_MACRO(Class_categories_120).csv'
results_root_path = f'./data/{dataset}/{evaluation}/'
plot_path = f'./plots/{dataset}/{evaluation}/'

res_dict = {} # will contain {'avg': {}, 'abs': {}, 'base': {}, 'cnn': {}}

for file_name in os.listdir(results_root_path):
    full_path = osp.join(results_root_path,file_name)
    res_dict[(file_name.split('_')[-1]).split('.')[0]] = torch.load(full_path, map_location='cpu', weights_only=False)

# Get the results dictionaries
avg = res_dict['avg']
abs = res_dict['abs']
base = res_dict['base']
cnn = res_dict['cnn']

# Using one, get a bincount of the true classes (are they balanced?)
# class_bins = base['truth'][0][:,-1].int()
true_bins = torch.bincount(base['truth'][0][:,-1].int())
pred_bins = torch.bincount(cnn['pred'][0][:,-2].int(), minlength=60)

pred_topk = torch.tensor([ # get the topk of the bincount of each video sample
    torch.topk(
        torch.bincount(sample.int()),
        k=1
    ).indices[0]
    for sample in cnn['pred'][0]
])

pred_topk = torch.bincount(pred_topk, minlength=60)

# Get the class names
all_classes = pd.read_csv(all_classes_path, delimiter=',', encoding='ISO-8859-1')
class_names = all_classes['Original class'][:60]


# GRAPHING
fig, ax = plt.subplots(figsize=(40, 20))

# ax.bar(np.linspace(0, 60, num=60), pred_topk, label=f"Predicted class frequency {model_type}")
ax.bar(np.arange(0, 60, 1), pred_bins, label=f"Predicted class frequency {model_type}")
ax.set_title(f"{model_type} class prediction frequency", fontsize=60)
ax.set_xlabel("Class number", fontsize=50)
ax.set_xlim(-1, 60)
ax.set_ylabel("Prediction frequency", fontsize=50)
ax.set_ylim(0, 360)

ax.xaxis.set_ticks(np.arange(0, 60, 1))
ax.tick_params(axis='both', labelsize=20)

ax.plot(np.linspace(0, 60, 60), np.ones(60)*275, '--', color='red', label="Median class distribution")

ax.legend(loc='lower right', fontsize=40)

plt.savefig(f"plots/ntu/CS/{model_type}_pred.png")

# classes_60_df = pd.read_csv(classes_60_path)


