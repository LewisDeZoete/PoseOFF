import matplotlib.pyplot as plt
import torch
import numpy as np

checkpoint_base_path = 'results/infogcn2/Flow_window5/infogcn_base.pt'
checkpoint_cnn_path = 'results/infogcn2/Flow_window5/infogcn_cnn.pt'
checkpoint_lin_path = 'results/infogcn2/Flow_window5/infogcn_linear.pt'
checkpoint_abs_path = 'results/infogcn2/Flow_window5/infogcn_abs_window_mean_flow.pt'
checkpoint_avg_path = 'results/infogcn2/Flow_window5/infogcn_average_flow.pt'
checkpoint_avg_c9_path = 'results/infogcn2/infogcn_avg.pt'
checkpoint_base = torch.load(checkpoint_base_path, map_location='cpu')
checkpoint_cnn = torch.load(checkpoint_cnn_path, map_location='cpu')
checkpoint_lin = torch.load(checkpoint_lin_path, map_location='cpu')
checkpoint_abs = torch.load(checkpoint_abs_path, map_location='cpu')
checkpoint_avg = torch.load(checkpoint_avg_path, map_location='cpu')
checkpoint_avg_c9 = torch.load(checkpoint_avg_c9_path, map_location='cpu')

results_base = checkpoint_base['results']
results_cnn = checkpoint_cnn['results']
results_lin = checkpoint_lin['results']
results_abs = checkpoint_abs['results']
# results_avg = checkpoint_avg['results'] FIXME: This checkpoint is missing
results_avg_c9 = checkpoint_avg_c9['results']

# Get the epochs as the x-axis
x = results_base['epoch']

# # ------------------------------
# #   Loss comparison graphing
# # ------------------------------

# # Plot the results
# fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25,7))
# fig.suptitle('Loss Comparison', fontsize=30)
# for ax in (ax1, ax2, ax3):
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.set_xlabel('Epoch', fontsize=20)
# ax1.set_ylabel('Loss', fontsize=20)

# # Plot class loss
# ax1.title.set_text('Classification Loss', )
# ax1.plot(x, results_base['train_cls_loss'], label='Base', color='tab:olive')
# ax1.plot(x, results_cnn['train_cls_loss'], label='CNN', color='tab:blue')
# ax1.plot(x, results_lin['train_cls_loss'], label='Linear', color='tab:red')
# ax1.plot(x, results_abs['train_cls_loss'], label='Flow magnitude averaged', color='tab:green')
# ax1.plot(x, results_avg['train_cls_loss'], label='Average flow direction (x,y)', color='tab:pink')
# ax1.legend()

# # Plot feature loss
# ax2.title.set_text('Feature Loss')
# ax2.plot(x, results_base['train_feature_loss'], label='Base', color='tab:olive')
# ax2.plot(x, results_cnn['train_feature_loss'], label='CNN', color='tab:blue')
# ax2.plot(x, results_lin['train_feature_loss'], label='Linear', color='tab:red')
# ax2.plot(x, results_abs['train_feature_loss'], label='Flow magnitude averaged', color='tab:green')
# ax2.plot(x, results_avg['train_feature_loss'], label='Average flow direction (x,y)', color='tab:pink')
# ax2.legend()

# # Plot reconstruction loss
# ax3.title.set_text('Reconstruction Loss')
# ax3.plot(x, results_base['train_recon_loss'], label='Base', color='tab:olive')
# ax3.plot(x, results_cnn['train_recon_loss'], label='CNN', color='tab:blue')
# ax3.plot(x, results_lin['train_recon_loss'], label='Linear', color='tab:red')
# ax3.plot(x, results_abs['train_recon_loss'], label='Flow magnitude averaged', color='tab:green')
# ax3.plot(x, results_avg['train_recon_loss'], label='Average flow direction (x,y)', color='tab:pink')
# ax3.legend()

# plt.savefig('Loss_comparison.png')


# # ------------------------------
# #   Accuracy comparison graphing
# # ------------------------------


# # Convert results to percentage
# def to_percentage(result):
#     return [r*100 for r in result]

# # Plot the results
# fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
# fig.suptitle('Accuracy Comparison', fontsize=30)
# for ax in (ax1, ax2):
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.set_xlabel('Epoch', fontsize=20)
#     ax.set_ylim(0, 100) # 0-100% accuracy
#     ax.set_xlim(0, 60)  # 0-60 epochs
#     # ax.axvline(x=40, color='black', linestyle='--', alpha=0.2, label='LR=0.01')  # Epoch 40
#     ax.axvline(x=50, color='black', linestyle='--', alpha=0.2, label='LR=0.01')  # Epoch 40
#     # ax.set_xscale('log')
# ax1.set_ylabel('Area Under Curve (Accuracy %)', fontsize=20)

# # Plot train AUC
# ax1.plot(x, to_percentage(results_base['train_AUC']), label='Base', color='tab:olive')
# ax1.plot(x, to_percentage(results_cnn['train_AUC']), label='CNN', color='tab:blue')
# ax1.plot(x, to_percentage(results_lin['train_AUC']), label='Linear', color='tab:red')
# ax1.plot(x, to_percentage(results_abs['train_AUC']), label='Flow magnitude averaged', color='tab:green')
# ax1.plot(x, to_percentage(results_avg['train_AUC']), label='Average flow direction (x,y)', color='tab:pink')
# ax1.set_title('Train AUC', fontsize=25)
# ax1.legend()

# # Plot test AUC
# ax2.plot(x, to_percentage(results_base['test_AUC']), label='Base', color='tab:olive')
# ax2.plot(x, to_percentage(results_cnn['test_AUC']), label='CNN', color='tab:blue')
# ax2.plot(x, to_percentage(results_lin['test_AUC']), label='Linear', color='tab:red')
# ax2.plot(x, to_percentage(results_abs['test_AUC']), label='Flow magnitude averaged', color='tab:green')
# ax2.plot(x, to_percentage(results_avg['test_AUC']), label='Average flow direction (x,y)', color='tab:pink')
# ax2.set_title('Test AUC', fontsize=22)
# # ax2.legend()

# plt.savefig('AUC_comparison.png')


# # ------------------------------
# #   Individual loss graphing
# # ------------------------------

# # Plot the results
# fig, ax = plt.subplots( figsize=(10, 10))
# ax.set_title('Classification loss - Base comparison\nTrain', fontsize=20)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax.set_xlabel('Epoch', fontsize=15)
# ax.set_ylabel('Loss', fontsize=15)
# # ax.set_ylim(0.0, 0.3) # 0-0.5 loss
# # ax.set_xlim(0, 60)  # 0-60 epochs
# # ax.set_xscale('log')


# statistic = 'cls_loss'


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

# ax.plot(x, base_comparison(results_cnn, statistic), label='CNN', color='tab:blue')
# ax.plot(x, base_comparison(results_lin, statistic), label='Linear', color='tab:red')
# ax.plot(x, base_comparison(results_abs, statistic), label='Flow magnitude averaged', color='tab:green')
# ax.plot(x, base_comparison(results_avg, statistic), label='Average flow direction (x,y)', color='tab:pink')
# ax.legend()

# # Adjust the bottom margin to make more space for the caption
# plt.subplots_adjust(bottom=0.15)

# caption = "Comparison of feature loss to base model.\nLoss=$\\frac{\\text{Model loss} - \\text{Base model loss}}{\\text{Base model loss}}$"
# fig.text(.5, .03, caption, ha='center', fontsize=10)

# plt.savefig('Classification_loss-base_comparison.png')


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
base_time = results_base['training_time']
cnn_time = results_cnn['training_time']
avg_c9_time = results_avg_c9['training_time']

running_avg_time = []
for time in avg_c9_time:
    try:
        if time < running_avg_time[-1]:
            running_avg_time.append(running_avg_time[-1]+time)
        else:
            running_avg_time.append(time)
    except IndexError:
        running_avg_time.append(time)
    

x2 = results_avg_c9['epoch']

# Plot the results
fig, ax = plt.subplots( figsize=(10, 10))
ax.plot(x, base_time, label='Base model', color='tab:olive')
ax.plot(x, cnn_time, label='CNN', color='tab:blue')
ax.plot(x2, running_avg_time, label='Average flow direction (x,y)', color='tab:pink')
ax.set_title('Training time comparison', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('Epoch', fontsize=15)
ax.set_ylabel('Time (s)', fontsize=15)

plt.savefig('Train_time_comparison.png')