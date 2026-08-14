### Load compare dict
import numpy as np
import torch
import shap

# Custom Imports
from utils.setup_args import SHAPArgs, create_shap_value_filepath, create_preds_savepath
import utils.shap_dataloader as sdl


# Select Algorithm and Dataset
algorithm = "der"
dataset = "cifar100"
shapArgs = SHAPArgs(algorithm, dataset)


first_last_only = True
filepath = create_shap_value_filepath(shapArgs, first_last_only) + ".npy"

num_tasks = shapArgs.dataset_params.num_task
num_class = shapArgs.dataset_params.num_class
cls_per_task = shapArgs.dataset_params.class_per_task
shap_samples = shapArgs.dataset_params.shap_samples


shap_values_loaded = np.load(filepath, allow_pickle=True)  # ['shap_dict']
#shap_values_loaded = np.load(f"analysis/{algorithm}/{dataset}/shap_values_first_last_1000.npy", allow_pickle=True)  # ['shap_dict']
num_imgs = len(shap_values_loaded[()].keys())
shap_dict = {}
for i in range(num_imgs):
    shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

'''
Bad Sample: 80
Good Sample: 132

Samples to Try: 55
'''

sample = 703
test_sample = shap_dict[f'{sample}']
test_sess = list(test_sample.keys())
test_sess.remove(test_sess[1])
print(test_sess)
ses, last_ses = int(test_sess[0][-1]), int(test_sess[1][-1])

'''
if dataset == "mnist":
    test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].reshape(28,28,1),
                  shap_dict[f'{sample}']['ses4']['shap_values'].reshape(28,28,1)]
else:
    test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].squeeze().reshape(32,32,3),
                  shap_dict[f'{sample}']['ses4']['shap_values'].squeeze().reshape(32,32,3)]
    print(test_shaps[0].shape)
'''

# Reshape/Transpose to Batch x Height x Width x Channel
if dataset == "mnist":
    test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].reshape(28,28,1),
                  shap_dict[f'{sample}'][f'ses{last_ses}']['shap_values'].reshape(28,28,1)]
else:
    test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].squeeze(-1).transpose([0, 2, 3, 1]),
                  shap_dict[f'{sample}'][f'ses{last_ses}']['shap_values'].squeeze(-1).transpose([0, 2, 3, 1])]

# Get test dataset
sal_dataloader = sdl.ShapDataloader(shapArgs)

if dataset == "cifar100":
    sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(ses * 10, (ses * 10) + 10), 20, batch_size=10000)
elif dataset == "imagenet200":
    sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(ses * 20, (ses * 20) + 20), 20, batch_size=10000)
else:
    ###---Updated to take initial classes learned into account---###
    if ses == 0:
        sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(shapArgs.dataset_params.init_cls),
                                                                      shapArgs.dataset_params.shap_samples, batch_size=10000)
    else:
        sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range((shapArgs.dataset_params.init_cls-cls_per_task)+ses*cls_per_task,
                                                                        shapArgs.dataset_params.init_cls+(ses*cls_per_task)),
                                                                        shapArgs.dataset_params.shap_samples, batch_size=10000)
    ###----------------------------------------------------------###
print("Len of sal_imgs:", len(sal_imgs))

test_imgs, test_labels = sal_imgs, sal_labels

# Get test image
samples = range(shap_samples*(num_class-cls_per_task))

adj_sample = sample - (ses * cls_per_task * shap_samples)

test_img = test_imgs[adj_sample]#.unsqueeze(0)
if dataset != "mnist":
    test_img = sal_dataloader.denormalize(test_img)
test_img_np = np.transpose(test_img.numpy(), [1, 2, 0])

labels = [f'ses{ses}', f'ses{num_tasks-1}']
shap.image_plot(np.concatenate(test_shaps), np.stack([test_img_np,test_img_np]), true_labels=labels)


# For a model with high SHAPC, but low accuracy:
# Look for a sample that was predicted correctly, feature consistency should be high
# Look for a sample that was predicted incorrectly, but check if feature consistency is still high -> indicates trustworthiness

