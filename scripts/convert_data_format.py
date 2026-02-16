import h5py
import os
import argparse
import numpy as np
from tqdm import tqdm

# ===================== Core Mapping Rules (Key: Complete this section later) =====================
# Mapping Rule Explanation:
# - key: Relative path under input traj_{idx} (e.g., "actions", "obs/sensor_data/base_camera_x/rgb")
# - value: Target path under output demo_{idx} + optional configurations
# - Format: {"old_rel_path": {"new_path": "new_rel_path", "dtype": target_type, "reshape": target_shape/processing_function}}
old_key_new_key_mapping = {
    "actions": {
        "new_path": "actions",
        "dtype": np.float32,
        "reshape": None,
    },
    "obs/sensor_data/base_camera_x/rgb": {
        "new_path": "obs/agentview_rgb",
        "dtype": np.uint8,
        "reshape": lambda x: x[:-1]
    },
    "obs/sensor_data/base_camera_x/segmentation": {
        "new_path": "obs/agentview_segmentation",
        "dtype": np.int32,
        "reshape": lambda x: x[:-1].repeat(2, axis=-1),
    },
    "obs/sensor_data/hand_camera/rgb": {
        "new_path": "obs/eye_in_hand_rgb",
        "dtype": np.uint8,
        "reshape": lambda x: x[:-1],
    },
    "obs/sensor_data/hand_camera/segmentation": {
        "new_path": "obs/eye_in_hand_segmentation",
        "dtype": np.int32,
        "reshape": lambda x: x[:-1].repeat(2, axis=-1),
    },
    "obs/extra/tcp_pose": {
        "new_path": "obs/tcp_pose",
        "dtype": np.float32,
        "reshape": lambda x: x[:-1]
    },
}

def get_traj_keys(h5_file):
    """
    Filter all trajectory keys in the format traj_{int} from the input HDF5 file and sort them numerically
    """
    traj_keys = []
    for key in h5_file.keys():
        if key.startswith("traj_") and key.split("_")[-1].isdigit():
            traj_keys.append(key)
    traj_keys.sort(key=lambda x: int(x.split("_")[-1]))
    return traj_keys

def process_single_traj(source_h5, traj_key, target_data_group, demo_idx):
    """
    Process a single trajectory, convert data from traj_key and write to demo_idx
    Parameters:
        source_h5: Opened input HDF5 file object (read-only)
        traj_key: Input trajectory name (e.g., traj_0)
        target_data_group: /data group object of the output file
        demo_idx: Output demo index (e.g., 0 corresponds to demo_0)
    """
    traj_group = source_h5[traj_key]
    demo_key = f"demo_{demo_idx}"
    if demo_key in target_data_group:
        del target_data_group[demo_key]  # Overwrite existing demo
    demo_group = target_data_group.create_group(demo_key)
    
    # Iterate through mapping rules and convert fields one by one
    for old_rel_path, config in old_key_new_key_mapping.items():
        # Skip unconfigured fields
        if config is None or "new_path" not in config:
            continue
        
        new_path = config["new_path"]
        target_dtype = config.get("dtype")
        reshape_func = config.get("reshape")

        data = traj_group[old_rel_path][...]
        
        # Shape adjustment
        if reshape_func is not None:
            if callable(reshape_func):
                data = reshape_func(data)
            else:  # Specific shape provided
                data = data.reshape(reshape_func)
        
        # Type conversion
        if target_dtype is not None:
            data = data.astype(target_dtype)
        
        # Write to target field (automatically create intermediate groups)
        # Split path and create intermediate groups (e.g., "obs/agentview_rgb" → create "obs" group)
        path_parts = new_path.split("/")
        current_group = demo_group
        for part in path_parts[:-1]:
            if part not in current_group:
                current_group = current_group.create_group(part)
            else:
                current_group = current_group[part]
        
        # Write data (overwrite existing field)
        if path_parts[-1] in current_group:
            del current_group[path_parts[-1]]
        current_group.create_dataset(
            path_parts[-1],
            data=data,
            dtype=data.dtype,
            compression="gzip"
        )
    
    return True, f"Trajectory {traj_key} → demo_{demo_idx} processed successfully"

def convert_maniskill_to_target(source_path, output_path, start_idx=None, end_idx=None):
    """
    Main conversion function: Convert ManiSkill format HDF5 to target format (/data/demo_{idx})
    Parameters:
        source_path: Path to input ManiSkill HDF5 file
        output_path: Path to output target HDF5 file
        start_idx: Starting trajectory index (optional, e.g., 0 means start from traj_0)
        end_idx: Ending trajectory index (optional, e.g., 100 means process up to traj_99)
    """
    if os.path.exists(output_path):
        print(f"Warning: Output file {output_path} already exists, will overwrite!")
        os.remove(output_path)
    
    with h5py.File(source_path, "r") as source_h5, h5py.File(output_path, "w") as target_h5:
        data_group = target_h5.create_group("data")
        
        traj_keys = get_traj_keys(source_h5)
        if not traj_keys:
            print("Error: No trajectory groups in the format traj_${int} found in input file")
            return
        
        # Filter trajectories by index (facilitate batch processing of large files)
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else len(traj_keys)
            traj_keys = traj_keys[start:end]
            print(f"Filtered trajectory range: {start} - {end}, total {len(traj_keys)} trajectories")
        
        # Batch process trajectories (traj_idx → demo_idx one-to-one correspondence)
        success_count = 0
        fail_count = 0
        for demo_idx, traj_key in enumerate(tqdm(traj_keys, desc="Processing trajectories")):
            success, msg = process_single_traj(source_h5, traj_key, data_group, demo_idx)
            if success:
                success_count += 1
            else:
                fail_count += 1
            print(msg)
        
        print(f"\nConversion completed: {success_count} succeeded, {fail_count} failed")
        print(f"Output file: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ManiSkill format H5 to LIBERO format")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--start", type=int, default=None, help="Starting trajectory index (optional, e.g., --start 0)")
    parser.add_argument("--end", type=int, default=None, help="Ending trajectory index (optional, e.g., --end 100)")
    args = parser.parse_args()
    
    convert_maniskill_to_target(args.input, args.output, args.start, args.end)