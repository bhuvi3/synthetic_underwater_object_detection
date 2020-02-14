# coding: utf8

"""
This runner script automates the pipeline for building yolo object detection model based on synthetic data,
by taking in a set of background images, the 3D model (egg file) of the object to be detected.
It evaluates the model on the given test data (which is supposed to be real data) whose ground truth is available.

"""

from PIL import Image

import argparse
import glob
import os
import shutil
import subprocess
import time

IMAGE_SIZE = 256

PYTHON = "/home/bhuvan/anaconda3/envs/tf2/bin/python"
CODE_DIR = "/home/bhuvan/Projects/underwater_synthetic_image_recognition/code"
DEFAULT_DARKNET_WEIGHTS_FILE = "/home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74"
DEFAULT_DARKNET_YOLO_CONFIG = "/home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg"

DARKNET_TRAIN_DIR_PATH = "/home/bhuvan/Projects/darknet"
DARKNET_EVAL_DIR_PATH  = "/home/bhuvan/Projects/darknet_AlexeyAB"


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
    parser.add_argument('--image-ext',
                        default="jpg",
                        help="The image extension of images in the background-dir. Default: 'jpg'.")
    parser.add_argument('--out-dir',
                        required=True,
                        help="The path to the intermediate files, trained models and evaluations need to be written.")
    parser.add_argument('--gt-images-dir',
                        required=True,
                        help="The path to the directory containing ground-truth images.")
    parser.add_argument('--gt-labels-dir',
                        required=True,
                        help="The path to the directory containing ground-truth labels.")

    args = parser.parse_args()

    return args


def run_cmd(cmd_list, log_file, check_success=True, no_write=False):
    cmd_list = list(map(str, cmd_list))

    if no_write:
        print("\n(no-write) Would run command: %s" % " ".join(cmd_list))
        return

    start_time = time.time()
    with open(log_file, "w") as log_file_fp:
        print("\nRunning command: %s" % " ".join(cmd_list))
        error_status = subprocess.run(cmd_list, stdout=log_file_fp, stderr=log_file_fp)

    if check_success and error_status.returncode != 0:
        raise ValueError("The command failed with the Error Status: %s." % error_status)

    print("Command completed: %s seconds." % (time.time() - start_time))


def resize_retain_aspect_ratio(src_dir, dest_dir, max_x_shape=256, max_y_shape=256, img_ext="jpg"):
    os.makedirs(dest_dir)
    size = max_x_shape, max_y_shape
    input_images = glob.glob(os.path.join(src_dir, "*.%s" % img_ext))
    for infile in input_images:
        im = Image.open(infile)
        im.thumbnail(size, Image.ANTIALIAS)
        outfile = os.path.join(dest_dir, os.path.basename(infile))
        im.save(outfile, img_ext)

    print("The images have been resized in the dest_dir: %s" % dest_dir)


def convert_greyscale_dir(src_dir, dest_dir, image_ext="jpg"):
    os.makedirs(dest_dir)
    src_images = glob.glob(os.path.join(src_dir, "*.%s" % image_ext))
    for src_img in src_images:
        img = Image.open(src_img).convert('L')
        img.save(os.path.join(dest_dir, os.path.basename(src_img)))

    print("The images have been converted to grayscale, and saved in the dest_dir: %s" % dest_dir)


def copy_label_files(src_dir, dest_dir, image_ext="jpg", label_ext="txt"):
    dest_images = glob.glob(os.path.join(dest_dir, ".%s" % image_ext))
    label_names = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], dest_images))
    for label_name in label_names:
        shutil.copy(os.path.join(src_dir,  "%s.%s" % (label_name, label_ext)),
                    os.path.join(dest_dir, "%s.%s" % (label_name, label_ext)))

    print("The labels have been copied from %s to %s." % (src_dir, dest_dir))


def run_synthetic_data_training(background_dir,
                                is_grayscale,
                                object_model_file,
                                num_train_scenes,
                                max_objects,
                                yolo_config,
                                pretrained_weights,
                                image_size,
                                image_ext,
                                out_dir,
                                gt_images_dir,
                                gt_labels_dir):
    def _create_abd_get_sub_out_dir(sub_dir_name):
        new_path = os.path.join(out_dir, sub_dir_name)
        os.makedirs(new_path)
        return new_path

    def _delete_intermediate_weights_files(weights_dir):
        intermediate_weights_files = glob.glob(os.path.join(weights_dir, "*0.weights"))
        for cur_weights_file in intermediate_weights_files:
            os.remove(cur_weights_file)

    # Create output directories.
    os.makedirs(out_dir)
    logs_dir = _create_abd_get_sub_out_dir("logs")
    rendered_data_dir = _create_abd_get_sub_out_dir("rendered_images")

    # Create a meta dict which holds the information of both synthetic and NST version.
    meta_dict = {"synthetic": {}, "nst": {}}
    data_formats = meta_dict.keys()

    # Step 1: Create Synthetic data by rendering the object on the background images (input original size images).
    print("\n# Step 1.")
    full_size_rendered_dir = os.path.join(rendered_data_dir, "synthetic-full_size")
    cmd_list = [
        PYTHON,
        os.path.join(CODE_DIR, "run_yolo_data_renderer.py"),
        "--background-dir", background_dir,
        "--object-model-file", object_model_file,
        "--num-scenes", num_train_scenes,
        "--max-objects", max_objects,
        "--out-dir", full_size_rendered_dir
    ]
    run_cmd(cmd_list, os.path.join(logs_dir, "run_yolo_data_renderer.log"), check_success=True)

    # Step 2: Resizing and converting to grayscale if required. Determining the content images.
    print("\n# Step 2.")
    resized_rendered_dir = os.path.join(rendered_data_dir, "synthetic")
    resize_retain_aspect_ratio(full_size_rendered_dir,
                               resized_rendered_dir,
                               max_x_shape=image_size,
                               max_y_shape=image_size,
                               img_ext=image_ext)
    copy_label_files(full_size_rendered_dir, resized_rendered_dir, image_ext="jpg", label_ext="txt")

    if not is_grayscale:
        meta_dict["synthetic"]["dataset_name"] = "synthetic"
        meta_dict["synthetic"]["data_dir"] = resized_rendered_dir
    else:
        resized_rendered_grayscale_dir = os.path.join(rendered_data_dir, "synthetic-gray")
        convert_greyscale_dir(resized_rendered_dir, resized_rendered_grayscale_dir, image_ext="jpg")
        copy_label_files(resized_rendered_dir, resized_rendered_grayscale_dir, image_ext="jpg", label_ext="txt")

        meta_dict["synthetic"]["dataset_name"] = "synthetic-gray"
        meta_dict["synthetic"]["data_dir"] = resized_rendered_grayscale_dir
        # Optionally: Delete the synthetic (color) images.

    # Step 3: Run NST on content images and style images (background_dir).
    print("\n# Step 3.")
    meta_dict["nst"]["dataset_name"] = meta_dict["synthetic"]["dataset_name"] + "-NST"
    meta_dict["nst"]["data_dir"] = meta_dict["synthetic"]["data_dir"] + "-NST"

    cmd_list = [
        PYTHON,
        os.path.join(CODE_DIR, "neural_style_transfer.py"),
        "--content-path", meta_dict["synthetic"]["data_dir"],
        "--style-path", background_dir,
        "--out-path", meta_dict["nst"]["data_dir"],
        "--max-dim", image_size
    ]
    run_cmd(cmd_list, os.path.join(logs_dir, "neural_style_transfer.log"), check_success=True)

    # Step 4: Create Yolo training files in the darknet dataset format, for each data_format.
    print("\n# Step 4.")
    yolo_training_dir = _create_abd_get_sub_out_dir("yolo_training_files")
    for data_format in data_formats:
        meta_dict[data_format]["darknet_dataset_path"] = os.path.join(yolo_training_dir,
                                                                      meta_dict[data_format]["dataset_name"])
        cmd_list = [
            PYTHON,
            os.path.join(CODE_DIR, "darknet_dataset_creator.py"),
            "--dataset-name", meta_dict[data_format]["dataset_name"],
            "--data-dir", meta_dict[data_format]["data_dir"],
            "--out-dir", meta_dict[data_format]["darknet_dataset_path"]
        ]
        run_cmd(cmd_list, os.path.join(logs_dir, "darknet_dataset_creator-%s.log" % data_format), check_success=True)

    # Step 5: Run Yolo training, for each data_format.
    print("\n# Step 5.")
    for data_format in data_formats:
        cmd_list = [
            os.path.join(DARKNET_TRAIN_DIR_PATH, "darknet"), "detector", "train",
            os.path.join(meta_dict[data_format]["darknet_dataset_path"], "%s.data" % meta_dict[data_format]["dataset_name"]),
            yolo_config,
            pretrained_weights,
        ]
        cur_yolo_log_file = os.path.join(meta_dict[data_format]["darknet_dataset_path"], "yolo_training.log")
        run_cmd(cmd_list, cur_yolo_log_file, check_success=True)
        print("Check the validation loss of the trained yolo model (%s) from the logs: %s"
              % (data_format, cur_yolo_log_file))
        print("Deleting the intermediate weight files to save disk memory from the backup dir: %s"
              % os.path.join(meta_dict[data_format]["darknet_dataset_path"], "darknet_backup"))
        _delete_intermediate_weights_files(os.path.join(meta_dict[data_format]["darknet_dataset_path"], "darknet_backup"))

    print("\n# Training Stage Complete.")

    # Step 6: Run Yolo detector on the test set.
    print("\n# Step 6.")
    for data_format in data_formats:
        cmd_list = [
            PYTHON,
            os.path.join(CODE_DIR, "run_yolo_detector.py"),
            "--input-dir", gt_images_dir,
            "--yolo-config-path", yolo_config,
            "--yolo-data-path", os.path.join(meta_dict[data_format]["darknet_dataset_path"],
                                             "%s.data" % meta_dict[data_format]["dataset_name"]),
            "--yolo-weights-path", os.path.join(meta_dict[data_format]["darknet_dataset_path"],
                                                "darknet_backup"
                                                "%s_final.weights" % os.path.splitext(os.path.basename(yolo_config))[0]),
            "--output-dir", os.path.join(meta_dict[data_format]["darknet_dataset_path"], "testset_detector_output"),
            "--image-ext", image_ext
        ]
        cur_detector_log_file = os.path.join(meta_dict[data_format]["darknet_dataset_path"], "testset_detector_output.log")
        run_cmd(cmd_list, cur_detector_log_file, check_success=True)

    # Step 7: Evaluate and compute mean Average Precision.
    print("\n# Step 7.")
    for data_format in data_formats:
        cmd_list = [
            PYTHON,
            os.path.join(CODE_DIR, "evaluate_detections.py"),
            "--gt-dir", gt_labels_dir,
            "--dr-dir", os.path.join(meta_dict[data_format]["darknet_dataset_path"],
                                     "testset_detector_output",
                                     "preds_voc_format"),
            "--images-dir", os.path.join(meta_dict[data_format]["darknet_dataset_path"],
                                         "testset_detector_output",
                                         "test_images_detected_labels"),
            "--output-dir", os.path.join(meta_dict[data_format]["darknet_dataset_path"],
                                         "testset_detector_output",
                                         "evaluation"),
            "--image-ext", image_ext
        ]
        cur_evaluation_log_file = os.path.join(meta_dict[data_format]["darknet_dataset_path"], "testset_evaluation.log")
        run_cmd(cmd_list, cur_evaluation_log_file, check_success=True)

    print("\n# Evaluation Stage Complete")


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
        args.image_ext,
        args.out_dir,
        args.gt_images_dir,
        args.gt_labels_dir
    )
