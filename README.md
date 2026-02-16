# [ICLR 2026] Translating Flow to Policy via Hindsight Online Imitation

[Yitian Zheng](https://github.com/unprintable123)\*,
[Zhangchen Ye](https://yzc0731.github.io/)\*,
[Weijun Dong](https://dwjshift.github.io/)\*,
[Shengjie Wang](https://shengjiewang-jason.github.io/),
[Yuyang Liu](https://scholar.google.com/citations?user=0ROQMVcAAAAJ&hl=en),
[Chongjie Zhang](https://mig-ai.github.io/person-zhangchongjie.html),
[Chuan Wen](https://alvinwen428.github.io/)<sup>&#x2709;</sup>,
[Yang Gao](https://yang-gao.weebly.com/)<sup>&#x2709;</sup>

[Paper](https://arxiv.org/abs/2512.19269) | [Website](https://dwjshift.github.io/HinFlow/)

![image](doc/hinflow.png)

## Installation

```sh
git clone --recursive git@github.com:yzc0731/HinFlow.git
cd HinFlow

conda env create -f environment.yml
conda activate hinflow

pip install -e third_party/robosuite/
pip install -e third_party/robomimic/
pip install -e third_party/maniskill/
```

## Dataset

We provide the preprocessed dataset to reproduce the results in our paper. You can download it from [Hugging Face Hub](https://huggingface.co/datasets/Zhangchen0731/HinFlow).

Or you can collect and preprocess the dataset yourself by following instructions below.

### Collect Dataset

For LIBERO tasks, you can download raw LIBERO dataset by running [download_libero_datasets](scripts/download_libero_datasets.py), do SpaceMouse teleoperation, or develop your own scripted policy. For more details, please refer to [CREATE YOUR OWN DATASETS](https://lifelong-robot-learning.github.io/LIBERO/html/tutorials/create_your_own_dataset.html) in LIBERO Docs.


For ManiSkill tasks, please refer to [ManiSkill Data Collection](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/index.html). Our method require control mode to be `pd_ee_delta_pose` and observation to be `rgb+segmentation`. 

Because the ManiSkill data format is different from LIBERO, we provide a script to convert [here](scripts/convert_data_format.py).

### Dataset Preprocessing

Dataset need to be preprocessed with [Cotracker](https://arxiv.org/abs/2307.07635) or [Cotracker3](https://arxiv.org/abs/2410.11831):

```sh
python -m scripts.preprocess \
  --source_hdf5=path/to/raw/data.hdf5 \
  --target_dir=path/to/preprocessed/data.hdf5 \
  --sampler=SegmentSampler \
  --use_points=1 \
  --sampler_cfg=path/to/preprocess/task.yaml \
  --env_type=maniskill
```

## Training

To replicate the results in our paper, use the following task names: `libero_butter`, `libero_book`, `libero_chocolate`, `libero_microwave`, `maniskill_pokecube`, `maniskill_pullcubetool`, and `maniskill_placesphere`. 

The training of our method includes two stages:

### Stage 1: High Level Planner

We have provided the checkpoints of High Level Planner to reproduce the results in our paper. You can download it from [Hugging Face Hub](https://huggingface.co/datasets/Zhangchen0731/HinFlow). Or you can do it yourself by following instructions below.

First, split the datasets into training and validation sets.

```sh
python -m scripts.split_trainval --folder=data/planner_dataset/${task}
```

The High Level Planner training can be executed by this command:

```sh
python -m scripts.train_planner --task=${task}
```

### Stage 2: Low Level Policy with Hindsight Online Imitation

Our policy can be trained with:

```sh
python -m scripts.train_hinflow_policy --task=${task} --gpu=${gpu_id} --planner=${planner_path}
```

Here `planner_path` is the path to the folder of the trained high level planner, it should contain `model_best.ckpt` and `config.yaml`.

## Baseline

To replicate the results in our paper, we provide 3 mode choices: `bc`, `atm_grid`, and `atm_seg`. The planner used in `atm_grid` and `atm_seg` baseline is the same as our method. In the training and evaluation of `bc`, `--planner` is required as a placeholder but will not be used.

Before training the baseline, process the dataset in `data/policy_dataset/${task}` using this script:

```sh
python -m scripts.label_points --task=${task} --mode=${mode}
```

Training scripts:

```sh
python -m scripts.train_baseline --task=${task} --planner=${planner_path} --mode=${mode}
```

Evaluation scripts:

```sh
python -m scripts.eval_baseline --task=${task} --exp-dir=path/to/your/exp/dir --planner=${planner_path} --mode=${mode}
```

## Citation

If you find our codebase is useful for your research, please cite our paper with this bibtex:

```
@inproceedings{zheng2026translating,
  title={Translating Flow to Policy via Hindsight Online Imitation},
  author={Zheng, Yitian and Ye, Zhangchen and Dong, Weijun and Wang, Shengjie and Liu, Yuyang and Zhang, Chongjie and Wen, Chuan and Gao, Yang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
