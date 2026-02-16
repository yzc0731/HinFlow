import os
import argparse
from glob import glob
import torch

# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task", 
                    choices=["libero_butter",
                             "libero_book",
                             "libero_chocolate",
                             "libero_microwave",
                             "maniskill_pokecube",
                             "maniskill_pullcubetool",
                             "maniskill_placesphere"])
args = parser.parse_args()

bench_name = args.task.split("_")[0]
task_name = args.task.split("_")[1]

if bench_name == "libero" and task_name == "microwave":
    views_list = ["agentview", "eye_in_hand_high"]
else:
    views_list = ["agentview", "eye_in_hand"]

CONFIG_NAME = f'{bench_name}_planner'
gpu_ids = list(range(torch.cuda.device_count()))
exp_name = f"{args.task}_planner"
EPOCH = 1001
root_dir = f'data/planner_dataset/{args.task}'
train_dataset_list = glob(os.path.join(root_dir, "*/train/"))
val1_dataset_list = glob(os.path.join(root_dir, "*/val/"))

command = (f'python -m engine.train_planner --config-name={CONFIG_NAME} '
           f'train_gpus="{gpu_ids}" '
           f'experiment={exp_name} '
           f'epochs={EPOCH} '
           f'dataset_cfg.views="{views_list}" '
           f'train_dataset="{train_dataset_list}" val_dataset="{val1_dataset_list}" ')

os.system(command)
