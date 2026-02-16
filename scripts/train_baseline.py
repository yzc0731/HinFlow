import os
import argparse
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
parser.add_argument("--planner", required=True, help="The path to the folder of high level planner.")
parser.add_argument("--mode", choices=["bc", "atm_grid", "atm_seg"])
parser.add_argument("--robot", default="Panda")
args = parser.parse_args()

train_gpu_ids = list(range(torch.cuda.device_count()))

root_dir = f'data/policy_dataset/{args.task}'
if args.mode == "bc":
    use_ground_truth_track = False
    use_zero_track = True
    repeat_sample_points = False
    augment_track = False
    spatial_transformer_use_language_token = True
    temporal_transformer_use_language_token = True
elif args.mode == "atm_grid":
    use_ground_truth_track = False
    use_zero_track = False
    repeat_sample_points = False
    augment_track = False
    spatial_transformer_use_language_token = False
    temporal_transformer_use_language_token = False
elif args.mode == "atm_seg":
    use_ground_truth_track = True
    use_zero_track = False
    repeat_sample_points = True
    augment_track = True
    spatial_transformer_use_language_token = False
    temporal_transformer_use_language_token = False

for seed in range(4):
    command = (f'python -m engine.train_baseline --config-name={args.task} train_gpus="{train_gpu_ids}" '
                f'+robot={args.robot} '
                f'experiment={args.mode}_policy_{args.task} '
                f'train_dataset="{root_dir}" val_dataset="{root_dir}" '
                f'model_cfg.track_cfg.track_fn={args.planner} '
                f'model_cfg.track_cfg.use_ground_truth_track={use_ground_truth_track} '
                f'model_cfg.track_cfg.use_zero_track={use_zero_track} '
                f'dataset_cfg.repeat_sample_points={repeat_sample_points} '
                f'dataset_cfg.augment_track={augment_track} '
                f'model_cfg.spatial_transformer_cfg.use_language_token={spatial_transformer_use_language_token} '
                f'model_cfg.temporal_transformer_cfg.use_language_token={temporal_transformer_use_language_token} '
                f'seed={seed} ')
    os.system(command)