# coding: utf8

"""
This runner script automates the pipeline for building yolo object detection model based on synthetic data,
by taking in a set of background images, the 3D model (egg file) of the object to be detected.
It evaluates the model on the given test data (which is supposed to be real data) whose ground truth is available.

"""

import argparse
import os
import shutil
import subprocess

IMAGE_SIZE = 256
DEFAULT_DARKNET_WEIGHTS_FILE = "/home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74"
DEFAULT_DARKNET_YOLO_CONFIG = "/home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg"
DARKNET_DIR_PATH = "/home/bhuvan/Projects/darknet_AlexeyAB"


def run_cmd(cmd_list, log_file, check_success=True):
    with open(log_file, "w") as log_file_fp:
        print("\nRunning command: %s" % " ".join(cmd_list))
        error_status = subprocess.run(cmd_list, stdout=log_file_fp, stderr=log_file_fp)

    if check_success and error_status.returncode != 0:
        raise ValueError("The command failed with the Error Status: %s." % error_status)


def get_args():
    parser = argparse.ArgumentParser(description=
    """
    This runner script automates the pipeline for building yolo object detection model based on synthetic data,
    by taking in a set of background images, the 3D model (egg file) of the object to be detected.
    It evaluates the model on the given test data (which is supposed to be real data) whose ground truth is available.
    """)
    parser.add_argument('--background-dir',
                        required=True,
                        help="The path to the directory containing the background images.")
    parser.add_argument('--is-grayscale',
                        action="store_true",
                        help="Specify if the images in the background-dir are grayscale images. Default: False.")
    parser.add_argument('--object-model-file',
                        required=True,
                        help="The path to the Panda3d object 3d-model file (.egg).")
    parser.add_argument('--num-train-scenes',
                        type=int,
                        required=True,
                        help="The number of scenes to generate out of the background images available for training.")
    parser.add_argument('--max-objects',
                        type=int,
                        default=2,
                        help="The maximum number of objects to be rendered on a scene.")
    parser.add_argument('--yolo-config',
                        default=DEFAULT_DARKNET_YOLO_CONFIG,
                        help="The path to the darknet YoloV3 config file."
                             "The training will be initialized with these weights. "
                             "Note that the image-size in the config file needs to match the image-size used here."
                             "Default: 'darknet53.conv.74' weights file from Darknet website would be used.")
    parser.add_argument('--pretrained-weights',
                        default=DEFAULT_DARKNET_WEIGHTS_FILE,
                        help="The path to the file containing the weights of trained YoloV3 (53) model (Darknet)."
                             "The training will be initialized with these weights, hence the Yolo config file must "
                             "define the same model architecture as the model which was used to produce these weights. "
                             "Default: 'darknet53.conv.74' weights file from Darknet website would be used.")
    parser.add_argument('--image-size',
                        type=int,
                        default=256,
                        help="The size of the image used for training Yolo models. "
                             "Provide the length of the side of the square image. Default: 256.")
    parser.add_argument('--out-dir',
                        required=True,
                        help="The path to the intermediate files, trained models and evaluations need to be written.")

    args = parser.parse_args()

    return args


def run_synthetic_data_training(background_dir,
                                is_grayscale,
                                object_model_file,
                                num_train_scenes,
                                max_objects,
                                yolo_config,
                                pretrained_weights,
                                image_size,
                                out_dir):
    pass


def run_testbed_analysis():
    pass


if __name__ == "__main__":
    args = get_args()

    # Generate synthetic scene data and train yolo models.
    run_synthetic_data_training(
        args.background_dir,
        args.is_grayscale,
        args.object_model_file,
        args.num_train_scenes,
        args.max_objects,
        args.yolo_config,
        args.pretrained_weights,
        args.image_size,
        args.out_dir
    )

    # Run yolo detector and evaluate on the testset provided.
    run_testbed_analysis()
