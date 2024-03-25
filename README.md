# Dataset Distillation via Knowledge Distillation: Towards Efficient Self-Supervised Pre-training of Deep Networks

Official repository for "Dataset Distillation via Knowledge Distillation: Towards Efficient Self-Supervised Pre-training of Deep Networks" which presents the method MKDT.

This code is based off of https://github.com/GeorgeCazenavette/mtt-distillation and https://github.com/wgcban/mix-bt.

## Setup

Clone the repository
```
git clone git@github.com:sjoshi804/mkdt-data-distill-ssl.git
```

Install the package 
```
pip install -e .
```

## Overview

1. Obtain representations from teacher model 
2. Train and save "expert trajectories" (referred to as buffers in code)
3. Distill dataset 
4. Evaluate by pre-training on distilled dataset and linear evaluation on downstream datasets

## Obtaining Representations from Teacher Model 

Any arbitrary model trained with SSL can be used to obtain the target representations. 
The only requirement is saving the representations for the dataset you wish to distill as a pytorch tensor, with the ith row corresponding to the representation of the ith example. 

For example, in the original paper we use the teacher models (trained using BarlowTwins) provided here: https://github.com/wgcban/mix-bt and extract the representations for a given dataset using the following command. 


```
python teacher_repr/get_teacher_repr.py --dataset <dataset> --batch_size <batch_size> --model <path_to_downloaded_model> --run_prefix <run_prefix> --device <gpu_id>

```

## Generating Expert Trajectories

Following MTT's code, we refer to expert trajectories as "buffers" in the code. 

```
python buffer.py --dataset=<dataset> --num_experts=<num_experts> --train_labels_path <train_labels_path> --buffer_path <path_to_save_buffers>
```

To parallelize runs, you can use the following script to run on gpus with ids from {start_device, ..., end_device}. 
Set num_runs s.t. num_experts * # gpus * num_runs = total number of desired buffers (expert trajectories)

```
./create_buffers.sh --dataset=<dataset>--num_experts=<num_experts> --train_labels_path= <train_labels_path> --start_device=0 --end_device=3 --num_runs=<num_runs> --env_name=<env_name> --save_dir=<save_dir>
```

P.S. For the aforementioned script, it is necessary to use arg_name=value convention for correct argument pasing.

## Distilling Dataset

```
CUDA_VISIBLE_DEVICES=GPU1,GPU2 python distill.py --dataset <dataset name> --train_labels_path <path_to_teacher_representations> --expert_epochs 2 --image_init_idx_path <path to pickle file containing indices of data to initialize distilled data>  --max_start_epoch 2 --expert_dir <path to expert trajectories / buffers folder> --iters <number of iters>
```

## Evaluating Distilled Dataset

Evaluate subset
```
python eval.py --subset_path <path to subset indices> --pre_epoch <pretraining_epochs> --train_dataset <dataset> --label_path <path_to_teacher_representations>
```

Evaluate distilled dataset-
```
python eval.py --result_dir <path to distilled dataset folder> --pre_epoch <pretraining_epochs> --train_dataset <dataset> --label_path <path_to_teacher_representations>
```

## Bibtex

```bibtex
n/a
```

## Steps to Update Dependencies 

```
pip install pip-tools
pip freeze > requirements.in
pip-compile requirements.in
pip-compile --output-file=- requirements.txt | pip-sync pyproject.toml
```