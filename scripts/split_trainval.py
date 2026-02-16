import argparse
import os
import glob
from natsort import natsorted

def split_pretrain_dataset(files, train_folder, val_folder, train_ratio):
    assert train_ratio == 0.9
    train_files = []
    test_files = []
    for i, f in enumerate(files):
        if i % 10 == 0:
            test_files.append(f)
        else:
            train_files.append(f)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for f in train_files:
        # Create relative symlinks
        os.system('ln -s -f {} {}'.format(os.path.relpath(f, train_folder), train_folder))

    for f in test_files:
        # Create relative symlinks
        os.system('ln -s -f {} {}'.format(os.path.relpath(f, val_folder), val_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    args = parser.parse_args()

    for task_name in os.listdir(args.folder):
        root_dir = os.path.join(args.folder, task_name)
        files = glob.glob(os.path.join(root_dir, '*.hdf5'))
        if len(files) == 0:
            raise ValueError('No .hdf5 files found in {}'.format(args.folder))

        files = natsorted(files)

        train_folder = os.path.join(root_dir, 'train')
        val_folder = os.path.join(root_dir, 'val')

        # if not os.path.exists(train_folder):
        split_pretrain_dataset(files, train_folder, val_folder, args.train_ratio)
