#!/usr/bin/python

"""
Creates the required files for training in Darknet.

# TODO: Support conversion of images from rgb to mono and resize etc.

"""

import argparse
import glob
import os
import random


def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Creates the files required for darknet training except the config file.

    """)
    parser.add_argument('--dataset-name',
                        required=True,
                        help="The name of the dataset.")
    parser.add_argument('--data-dir',
                        required=True,
                        help="The path to the directory containing darknet dataset where the images and label files"
                             "are present.")
    parser.add_argument('--classes',
                        required=True,
                        help="The names of the classes as a comma-separated string.")
    parser.add_argument('--out-dir',
                        required=True,
                        help="The path to the directory to which the darknet training files need to be written.")
    parser.add_argument('--val-split',
                        type=float,
                        default=0.20,
                        help="The proportion of validation data to the whole data. Default: 0.20.")
    parser.add_argument('--image-ext',
                        default="jpg",
                        help="The extension of the image files in the --data-dir. Default: 'jpg'.")

    args = parser.parse_args()
    args.classes = args.classes.split(",")

    return args


def create_darknet_training_files(dataset_name, data_dir, classes, val_split, out_dir, image_ext):
    # 1. Create Train and Validation files.
    data_dir = os.path.abspath(data_dir)
    image_paths = glob.glob(os.path.join(data_dir, "*.%s" % image_ext))

    random.shuffle(image_paths)  # Shuffling the order (in-place).

    total_count = len(image_paths)
    val_count = int(val_split * total_count)
    print("Found %s images out of which %s images will be considered as validation set (%s proportion)"
          % (total_count, val_count, val_split))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_file_path = os.path.join(out_dir, "%s-train.txt" % dataset_name)
    val_file_path = os.path.join(out_dir, "%s-val.txt" % dataset_name)
    names_file_path = os.path.join(out_dir, "%s.names" % dataset_name)
    data_file_path = os.path.join(out_dir, "%s.data" % dataset_name)
    backup_dir_path = os.path.join(out_dir, "darknet_backup")
    os.makedirs(backup_dir_path)

    for x in [train_file_path, val_file_path, names_file_path, data_file_path]:
        if os.path.exists(x):
            raise ValueError("The darknet training file exists in the out-dir: %s" % x)

    with open(train_file_path, "w") as train_fp, open(val_file_path, "w") as val_fp:
        c = 0
        for image_path in image_paths:
            c += 1
            if c <= val_count:
                val_fp.write("%s\n" % image_path)
            else:
                train_fp.write("%s\n" % image_path)


    # 2. Create .names file.
    with open(names_file_path, "w") as names_fp:
        for class_name in classes:
            names_fp.write("%s\n" % class_name)

    # 3. Create .data file.
    with open(data_file_path, "w") as data_fp:
        data_fp.write("classes=%s\n" % len(classes))
        data_fp.write("train=%s\n" % os.path.abspath(train_file_path))
        data_fp.write("val=%s\n" % os.path.abspath(val_file_path))
        data_fp.write("names=%s\n" % os.path.abspath(names_file_path))
        data_fp.write("backup=%s\n" % os.path.abspath(backup_dir_path))

    print("The darknet training files have been created at %s" % os.path.abspath(out_dir))
    print("Manual Task: Copy the darknet config to this directory and update the values.")


if __name__ == "__main__":
    args = get_args()
    create_darknet_training_files(args.dataset_name,
                                  args.data_dir,
                                  args.classes,
                                  args.val_split,
                                  args.out_dir,
                                  args.image_ext)
