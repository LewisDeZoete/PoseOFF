import matplotlib.pyplot as plt
import torch

checkpoint_base_path = 'results/infogcn2/infogcn_base.pt'
checkpoint_cnn_path = 'results/infogcn2/infogcn_cnn.pt'
checkpoint_lin_path = 'results/infogcn2/infogcn_linear.pt'
checkpoint_abs_path = 'results/infogcn2/infogcn_abs_window_mean_flow.pt'
checkpoint_avg_path = 'results/infogcn2/infogcn_average_flow.pt'
checkpoint_base = torch.load(checkpoint_base_path, map_location='cpu')
checkpoint_cnn = torch.load(checkpoint_cnn_path, map_location='cpu')
checkpoint_lin = torch.load(checkpoint_lin_path, map_location='cpu')
checkpoint_abs = torch.load(checkpoint_abs_path, map_location='cpu')
checkpoint_avg = torch.load(checkpoint_avg_path, map_location='cpu')
results_base = checkpoint_base['results']
results_cnn = checkpoint_cnn['results']
results_lin = checkpoint_lin['results']
results_abs = checkpoint_abs['results']
results_avg = checkpoint_avg['results']

print(results_base.keys())

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
# ax1.set_ylabel('Area Under Curve (Accuracy %)', fontsize=20)

# # Plot train AUC
# ax1.plot(x, to_percentage(results_base['train_AUC']), label='Base train AUC', color='tab:olive')
# ax1.plot(x, to_percentage(results_cnn['train_AUC']), label='CNN train AUC', color='tab:blue')
# ax1.plot(x, to_percentage(results_lin['train_AUC']), label='Linear train AUC', color='tab:red')
# ax1.plot(x, to_percentage(results_abs['train_AUC']), label='Flow magnitude averaged AUC', color='tab:green')
# ax1.plot(x, to_percentage(results_avg['train_AUC']), label='Average flow direction (x,y) AUC', color='tab:pink')
# ax1.legend()

# # Plot test AUC
# ax2.plot(x, to_percentage(results_base['test_AUC']), label='Base train AUC', color='tab:olive')
# ax2.plot(x, to_percentage(results_cnn['test_AUC']), label='CNN test AUC', color='tab:blue')
# ax2.plot(x, to_percentage(results_lin['test_AUC']), label='Linear test AUC', color='tab:red')
# ax2.plot(x, to_percentage(results_abs['test_AUC']), label='Flow magnitude averaged AUC', color='tab:green')
# ax2.plot(x, to_percentage(results_avg['test_AUC']), label='Average flow direction (x,y) AUC', color='tab:pink')
# ax2.legend()

# plt.savefig('AUC_comparison.png')


# ------------------------------
#   Individual loss graphing
# ------------------------------

# Plot the results
fig, ax = plt.subplots( figsize=(10, 10))
ax.set_title('Reconstruction loss', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel('Epoch', fontsize=20)

statistic = 'cls_loss'
ax.set_ylabel('Loss', fontsize=20)

# Plot reconstruction loss
ax.plot(x, results_base[f'train_{statistic}'], label='Base model', color='tab:olive')
ax.plot(x, results_cnn[f'train_{statistic}'], label='CNN', color='tab:blue')
ax.plot(x, results_lin[f'train_{statistic}'], label='Linear', color='tab:red')
ax.plot(x, results_abs[f'train_{statistic}'], label='Flow magnitude averaged', color='tab:green')
ax.plot(x, results_avg[f'train_{statistic}'], label='Average flow direction (x,y)', color='tab:pink')
ax.legend()

plt.savefig('Classification_loss.png')