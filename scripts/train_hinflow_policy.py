import os
import argparse
import random

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
parser.add_argument("--gpu", default="0", help="The gpu id used for training.")
parser.add_argument("--planner", required=True, help="The path to the folder of high level planner.")
parser.add_argument("--robot", default="Panda")
parser.add_argument("--sampler", default="SegmentSampler")
parser.add_argument("--agent", default="hinflow")
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

train_gpu_ids = [0]
command = (f'python -m engine.train_hinflow_policy --config-name={args.task} train_gpus="{train_gpu_ids}" '
            f'dry=true '
            f'+robot={args.robot} '
            f'++fabric_device_idx={args.gpu} '
            f'+sampler={args.sampler} '
            f'experiment=hinflow_policy_{args.task} '
            f'agent_cfg={args.agent} '
            f'agent_cfg.model_cfg.track_cfg.track_fn={args.planner} '
            f'env_cfg.env_name="{suite_name_list}" env_cfg.task_name="{task_name_list}" '
            f'seed={random.randint(1, 9999)} ')

os.system(command)