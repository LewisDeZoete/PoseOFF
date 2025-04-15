import matplotlib.pyplot as plt
import torch
import numpy as np
import os.path as osp

# dataset = 'nturgbd'
# evaluation = 'CS'
dataset = 'ucf101'
evaluation = ''

if dataset == 'ucf101':
    checkpoint_base_path = f'results/{dataset}/infogcn_base.pt'
    checkpoint_abs_path = f'results/{dataset}/infogcn_abs.pt'
    checkpoint_avg_path = f'results/{dataset}/infogcn_avg.pt'
    checkpoint_cnn_path = f'results/{dataset}/infogcn_cnn.pt'
else:
    checkpoint_base_path = f'results/{dataset}/{evaluation}/{dataset}_{evaluation}_base.pt'
    checkpoint_abs_path = f'results/{dataset}/{evaluation}/{dataset}_{evaluation}_abs.pt'
    checkpoint_avg_path = f'results/{dataset}/{evaluation}/{dataset}_{evaluation}_avg.pt'
    checkpoint_cnn_path = f'results/{dataset}/{evaluation}/{dataset}_{evaluation}_cnn.pt'
checkpoint_base = torch.load(checkpoint_base_path, map_location='cpu')
checkpoint_abs = torch.load(checkpoint_abs_path, map_location='cpu')
checkpoint_avg = torch.load(checkpoint_avg_path, map_location='cpu')
checkpoint_cnn = torch.load(checkpoint_cnn_path, map_location='cpu')

results_base = checkpoint_base['results']
results_abs = checkpoint_abs['results']
results_avg = checkpoint_avg['results']
results_cnn = checkpoint_cnn['results']

# Get the epochs as the x-axis
x = results_base['epoch']

# Evaluation string and Dataset string dictionaries for plot titles
eval_str_dict = {'CS': 'Cross Subject', 'CV': 'Cross View'}
dataset_str_dict = {'nturgbd': 'NTU RGB+D', 'nturgbd120': 'NTU RGB+D 120', 'ucf101':'UCF-101'}
if dataset == 'ucf101':
    fig_title = f'{dataset_str_dict[dataset]}'
    save_root = f'results/plots/{dataset}/'
    print(f'Plotting results for {dataset_str_dict[dataset]}')
else:
    fig_title = f'{dataset_str_dict[dataset]} - {eval_str_dict[evaluation]}'
    save_root = f'results/plots/{dataset}-{evaluation}/'
    print(f'Plotting results for {dataset_str_dict[dataset]} - {eval_str_dict[evaluation]} evaluation')

# # ------------------------------
# #   Loss comparison graphing
# # ------------------------------

# # Plot the results
# fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25,7))
# fig.suptitle(f'Loss Comparison - {fig_title}', fontsize=30)
# for ax in (ax1, ax2, ax3):
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.set_xlabel('Epoch', fontsize=20)
# ax1.set_ylabel('Loss', fontsize=20)

# # Plot class loss
# ax1.set_title('Classification Loss', fontsize = 25)
# ax1.plot(x, results_base['train_cls_loss'], label='Base', color='tab:olive')
# ax1.plot(x, results_abs['train_cls_loss'], label='Absolute flow', color='tab:green')
# ax1.plot(x, results_avg['train_cls_loss'], label='Average flow', color='tab:pink')
# ax1.plot(x, results_cnn['train_cls_loss'], label='CNN learning', color='tab:blue')
# ax1.legend()

# # Plot feature loss
# ax2.set_title('Feature Loss', fontsize = 25)
# ax2.plot(x, results_base['train_feature_loss'], label='Base', color='tab:olive')
# ax2.plot(x, results_abs['train_feature_loss'], label='Absolute flow', color='tab:green')
# ax2.plot(x, results_avg['train_feature_loss'], label='Average flow', color='tab:pink')
# ax2.plot(x, results_cnn['train_feature_loss'], label='CNN', color='tab:blue')

# # Plot reconstruction loss
# ax3.set_title('Reconstruction Loss', fontsize = 25)
# ax3.plot(x, results_base['train_recon_loss'], label='Base', color='tab:olive')
# ax3.plot(x, results_abs['train_recon_loss'], label='Average flow', color='tab:green')
# ax3.plot(x, results_avg['train_recon_loss'], label='Average flow', color='tab:pink')
# ax3.plot(x, results_cnn['train_recon_loss'], label='CNN', color='tab:blue')

# plt.tight_layout()
# plt.savefig(osp.join(save_root,'loss_comparison/Loss_comparison.png'))


# ------------------------------
#   Accuracy comparison graphing
# ------------------------------

# Convert results to percentage
def to_percentage(result):
    return [r*100 for r in result]

# Plot the results
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
fig.suptitle(f'Accuracy Comparison - {fig_title}', fontsize=30)
for ax in (ax1, ax2):
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylim(0, 100) # 0-100% accuracy
    ax.set_xlim(0, 70)  # 0-70 epochs
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.2, label='LR=0.01')  # Epoch 40
    ax.axvline(x=60, color='black', linestyle='--', alpha=0.2, label='LR=0.001')  # Epoch 40
    # ax.set_xscale('log')
ax1.set_ylabel('Classification Accuracy %', fontsize=20)

# Plot train Accuracy
ax1.plot(x, to_percentage(results_base['train_AUC']), label='Base', color='tab:olive')
ax1.plot(x, to_percentage(results_abs['train_AUC']), label='Absolute flow', color='tab:green')
ax1.plot(x, to_percentage(results_avg['train_AUC']), label='Average flow', color='tab:pink')
ax1.plot(x, to_percentage(results_cnn['train_AUC']), label='CNN', color='tab:blue')
ax1.set_title('Train accuracy', fontsize=25)
ax1.legend()

# Plot test Accuracy
ax2.plot(x, to_percentage(results_base['test_AUC']), label='Base', color='tab:olive')
ax2.plot(x, to_percentage(results_abs['test_AUC']), label='Absolute flow', color='tab:green')
ax2.plot(x, to_percentage(results_avg['test_AUC']), label='Average flow', color='tab:pink')
ax2.plot(x, to_percentage(results_cnn['test_AUC']), label='CNN', color='tab:blue')
ax2.set_title('Test accuracy', fontsize=25)

for name, results in {'Base': results_base, 
                      'Absolute': results_abs, 
                      'Average': results_avg, 
                      'CNN': results_cnn}.items():
    print(f"{name}: {max(to_percentage(results['test_AUC'])):.2f}%")

plt.savefig(osp.join(save_root,'accuracy/Acc_comparison.png'))

# # ------------------------------
# #   Individual loss graphing
# # ------------------------------

# statistics = ['cls_loss', 'recon_loss', 'feature_loss']
# statistic_str_dict = {'cls_loss': 'Classification Loss', 
#                  'recon_loss': 'Reconstruction Loss',
#                  'feature_loss': 'Feature Loss'}


# # Plot reconstruction loss
# # ax.plot(x, results_base[f'train_{statistic}'], label='Base model', color='tab:olive')
# # ax.plot(x, results_cnn[f'train_{statistic}'], label='CNN', color='tab:blue')
# # ax.plot(x, results_lin[f'train_{statistic}'], label='Linear', color='tab:red')
# # ax.plot(x, results_abs[f'train_{statistic}'], label='Flow magnitude averaged', color='tab:green')
# # ax.plot(x, results_avg[f'train_{statistic}'], label='Average flow direction (x,y)', color='tab:pink')
# # ax.legend()

# def base_comparison(results_dict, statistic: str, state: str = 'train'):
#     comparison = (np.array(results_dict[f'{state}_{statistic}']) - np.array(results_base[f'{state}_{statistic}'])) / np.array(results_base[f'{state}_{statistic}'])
#     return comparison

# for statistic in statistics:
#     # Plot each statistic's loss with comparison to base
#     fig, ax = plt.subplots( figsize=(10, 10))
#     # ax.set_title(f'{statistic_str_dict[statistic]} - Base comparison (Train)\n{dataset_str_dict[dataset]} - {eval_str_dict[evaluation]}', fontsize=20)
#     ax.set_title(f'{statistic_str_dict[statistic]} - Base comparison (Train)\n{dataset_str_dict[dataset]}', fontsize=20) # UCF101
#     ax.tick_params(axis='both', which='major', labelsize=15)
#     ax.set_xlabel('Epoch', fontsize=15)
#     ax.set_ylabel('Loss', fontsize=15)
#     # ax.set_ylim(0.0, 0.3) # 0-0.5 loss
#     # ax.set_xlim(0, 60)  # 0-60 epochs
#     # ax.set_xscale('log')
#     ax.plot(x, base_comparison(results_cnn, statistic), label='CNN', color='tab:blue')
#     ax.plot(x, base_comparison(results_abs, statistic), label='Flow magnitude averaged', color='tab:green')
#     ax.plot(x, base_comparison(results_avg, statistic), label='Average flow direction (x,y)', color='tab:pink')
#     ax.hlines(0, 0, 70, colors='black', linestyles='--', label='Base model loss', color='tab:olive')
#     ax.legend()

#     # Adjust the bottom margin to make more space for the caption
#     plt.subplots_adjust(bottom=0.15)

#     caption = f"Comparison of {statistic_str_dict[statistic].lower()} to base model." + \
#     "\nLoss=$\\frac{\\text{Model loss} - \\text{Base model loss}}{\\text{Base model loss}}$"
#     fig.text(.5, .03, caption, ha='center', fontsize=10)

#     # plt.savefig(f'results/plots/{dataset}-{evaluation}/loss_individual/{statistic}-base_comparison.png')
#     plt.savefig(f'results/plots/{dataset}/loss_individual/{statistic}-base_comparison.png')


# # ------------------------------
# #   conf matrix testing
# # ------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# # Assuming y_true and y_pred are your true labels and predicted labels
# y_true = np.random.randint(0, 101, size=1000)  # Example data
# y_pred = np.random.randint(0, 101, size=1000)  # Example data

# cm = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(20, 20))
# sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()


# # ------------------------------
# # Training time
# # ------------------------------