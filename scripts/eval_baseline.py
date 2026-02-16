import os
import argparse

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
parser.add_argument("--exp-dir", required=True, help="The path to the folder of trained policy.")
parser.add_argument("--planner", required=True, help="The path to the folder of high level planner.")
parser.add_argument("--mode", choices=["bc", "atm_grid", "atm_seg"])
parser.add_argument("--robot", default="Panda")
parser.add_argument("--gpu", default="0", help="The gpu id used for training.")
args = parser.parse_args()

task_name = args.task.split("_")[1]
task_dict = {
    "butter": ["pick_up_the_butter_and_place_it_in_the_basket"], 
    "book": ["STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy"],
    "chocolate": ["KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it"],
    "microwave": ["KITCHEN_SCENE6_close_the_microwave"],
    "pokecube": "PokeCube-v2", 
    "pullcubetool": "PullCubeTool-v2",
    "placesphere": "PlaceSphere-v2",
}
task_name_list = task_dict[task_name]

suite_dict = {
    "butter": "libero_test",
    "book": "libero_90",
    "chocolate": "libero_90",
    "microwave": "libero_90",
    "pokecube": "maniskill",
    "pullcubetool": "maniskill",
    "placesphere": "maniskill",
}
suite_name = suite_dict[task_name]
suite_name_list = [suite_name]

# evaluation configs
train_gpu_ids = [int(args.gpu)]
env_gpu_ids = [int(args.gpu)]

if args.mode == "bc" or args.mode == "atm_grid":
    sampler_type = "GridSampler"
elif args.mode == "atm_seg":
    sampler_type = "SegmentSampler"

command = (f'python -m engine.eval_baseline --config-dir={args.exp_dir} --config-name=config hydra.run.dir=/tmp '
            f'+save_path={args.exp_dir} '
            f'train_gpus="{train_gpu_ids}" '
            f'++robot={args.robot} '
            f'+sampler={sampler_type} '
            f'env_cfg.env_name="{suite_name_list}" '
            f'env_cfg.task_name="{task_name_list}" '
            f'model_cfg.track_cfg.track_fn="{args.planner}" '
            f'env_cfg.render_gpu_ids="{env_gpu_ids}" env_cfg.vec_env_num=1 ')

os.system(command)
