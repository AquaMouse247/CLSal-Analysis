# Improved Efficiency and Class Dependency Examination with SHAPC
## Introduction
This repository contains code for three optimized approaches to computing SHAPC-Mean that reduce samples, task-pair iterations, or a combination of both. This decreases overall computation time while maintaining metric performance. Additionally, it contains code for calculating a classwise SHAPC-Mean metric to assess potential class-dependent biases.

## Results
## How To Use
<!--### Dependencies
The following Python libraries are required:
run `pip install -r requirements.txt` to install all dependencies.
-->

### Clone
Clone this GitHub repository:
```
git clone https://github.com/AquaMouse247/CL-SHAPC-Interpretability.git
cd CL-SHAPC-Interpretability
```

### Requirements
In order to run the experiments, a minimum of Python 3.10.0 is required, along with the following packages:
```
kornia==0.8.2
numpy==2.4.4
onedrivedownloader==1.1.3
Pillow==12.2.0
quadprog==0.1.13
scikit_learn==1.8.0
scipy==1.17.1
shap==0.48.0
torch==2.7.0
torchvision==0.22.0
tqdm==4.66.5
wandb==0.26.1
```

In order to run the MATLAB scripts, a minimum of MATLAB R2024b is required.

### Datasets
The code can generate results for the following datasets:
- CIFAR10
- CIFAR100
- TinyImageNet

### Algorithms
A total of nine algorithms were tested:
- `iTAML`
- `RPSnet`
- `FOSTER`
- `MEMO`
- `DER`
- `iCARL`
- `DS-AL`
- `TagFex`
- `XDER`

### Saved Models
This code relies on saved models. In order to load these models, the `load_model.py` script must be used. Inside this script, set the `filepath` parameter to the root folder for your saved models. To use a specific model, put the saved model.pth file into the `[root_model_filepath]/[algorithm]/[saved_model].pth`.

## Running Experiments

### Generating SHAPC Values
First the `generate_shap_values.py` script should be used to generate SHAP values.
resulting in a `shap_values_first_last_[samples_number].npy` file in `analysis/[algorithm]/[dataset]`.

Then the `generate_shapc_values.py` script should be used resulting in a `shapc_values_first_last_[samples_number].mat` file in `analysis/[algorithm]/[dataset]`.
***

### Generating SHAPC-Mean Values
The `generate_shapc_mean_values.m` script should be used resulting in the SHAPC-Mean value being output in the console.
***

### Identifying Scenarios
**Requires the SHAPC values for the desired algorithms and datasets.**

1. Run the `generate_shap_preds.py` script to get the predictions for the associated SHAP images for each algorithm and dataset to receive a `[algorithm]_[dataset]_preds.mat` file in `analysis/preds/[algorithm]/[dataset]`.
2. Run the `merge_preds.mlx` script to generate/update the `cifar10_preds.mat` file for each desired algorithm and dataset.
3. Use the `identify_scenarios.mlx` script to receive the FoM tables.

## Acknowledgments
## Citation
