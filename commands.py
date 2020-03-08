# Commands.

# Docker:
docker run --gpus all -itd -p 8888:8888 -p 16006:16006 tensorflow/tensorflow:latest-gpu-py3 bash

docker exec -ti a0bb7eeed020 bash

docker stop a0bb7eeed020 ; docker rm a0bb7eeed020

jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

docker cp path_in_local a0bb7eeed020:/projects/


# GPU real-time monitor.
watch -n0.1 nvidia-smi


# All scripts run from the code directory.

### Create data.
# Yolo data synthetic renderer.
python run_yolo_data_renderer.py --background-dir ../data/underwater_background/unsplash/unsplash_underwater_collection --object-model-file ../panda3d_models/mine.egg --num-scenes 1000 --out-dir ../data/darknet_datasets/unsplash_mine_raw/darknet_images_labels --max-objects 2

# Manual: Check image names.

# Draw bounding box and verify labels.
python ../../../code/show_bounding_boxes.py ../unsplash_mine_raw_test/darknet_images_labels-sleep_0.001-NST
python show_bounding_boxes.py ../data/darknet_datasets/unsplash_mine_raw/darknet_images_labels

# Neural Style Transfer
# single image:
python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw_test/darknet_images_labels-sleep_0.001/adam-eperjesi-et7JPPrMtIw-unsplash-1.jpg --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection/adam-eperjesi-et7JPPrMtIw-unsplash.jpg --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw_test/darknet_images_labels-sleep_0.001-NST_test/adam-eperjesi-et7JPPrMtIw-unsplash-1.jpg

# multiple images (images-labels-dir)
python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw_test/darknet_images_labels-sleep_0.001 --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw_test/darknet_images_labels-sleep_0.001-NST

python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-NST --grayscale

### Darknet
# Data preparation:
python darknet_dataset_creator.py --dataset-name unsplash_rendered_test --data-dir ../data/darknet_datasets/unsplash_rendered_test/darknet_images_labels --classes torus --out-dir ../data/darknet_datasets/unsplash_rendered_test

# Manually add the config file like ("yolov3-tiny.cfg"), and have the pretrained weights from "darknet53.conv.74". Set max_batches to epochs/

# Training.
./darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_rendered_test/unsplash_rendered_test.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_rendered_test/yolov3-tiny.cfg darknet53.conv.74

# Testing.
./darknet detector test /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_rendered_test/unsplash_rendered_test.data  /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_rendered_test/yolov3-tiny.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_rendered_test/darknet_backup/yolov3-tiny_last.weights /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_rendered_test/darknet_images_labels/almos-bechtold-A6TVJBNA-z4-unsplash-107.jpg

###
# Correcting segmentation fault:
deleted some images which did not have matching labels (5), the corrected darknet_images_labels: /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels

# TODO: Review the neural_style_transfer script which is still unstaged.

# TODO: Check yolo3-wells.cfg from unsplash_rendered_test

# previous run:
python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-NST --grayscale --max-dim 256 &> nst_run2.log


# Resizing the images in the unsplash_mine_raw/darknet_images_labels (original images stored in unsplash_mine_raw/darknet_images_labels-orig).
from PIL import Image
import glob
import os
import shutil

size = 256, 256

input_images = glob.glob("darknet_images_labels-orig/*.jpg")
for infile in input_images:
    outfile = os.path.join("darknet_images_labels", os.path.basename(infile))
    im = Image.open(infile)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(outfile, "JPEG")

label_files = glob.glob("darknet_images_labels-orig/*.txt")
for label_file in label_files:
    shutil.copy(label_file, os.path.join("darknet_images_labels", os.path.basename(label_file)))


# Converting color images to greyscale in a directory.
from PIL import Image

import glob
import os
import shutil

def _convert_greyscale_dir(src_dir, dest_dir):
    os.makedirs(dest_dir)
    src_images = glob.glob(os.path.join(src_dir, "*.jpg"))
    for src_img in src_images:
        img = Image.open(src_img).convert('L')
        img.save(os.path.join(dest_dir, os.path.basename(src_img)))

# Convert the source unsplash images to greyscale, and convert the rendered images (unsplash_mine_raw/darknet_images_labels).
_convert_greyscale_dir("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection",
                       "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection-gray")


_convert_greyscale_dir("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels",
                       "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-gray")


label_files = glob.glob("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels/*.txt")
for label_file in label_files:
    shutil.copy(label_file, os.path.join("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-gray", os.path.basename(label_file)))



_convert_greyscale_dir("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-sample",
                       "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-sample-gray")

label_files = glob.glob("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-sample/*.txt")
for label_file in label_files:
    shutil.copy(label_file, os.path.join("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-sample-gray", os.path.basename(label_file)))




# Runnning NST on sample dataset.
python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-sample --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-sample-NST --grayscale --max-dim 256 &> ../logs/nst_run-sample.log


python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-sample-gray --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection-gray --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-sample-gray-NST --max-dim 256 &> ../logs/nst_run-sample_gray.log


# Running NST on resized images.
python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-NST --grayscale --max-dim 256 &> ../logs/nst_run-resized.log

# Renaming files:
sample files moved to sample_files directory.
"darknet_images_labels" has been changed to "synthetic".

python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection-gray --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST --max-dim 256 &> ../logs/nst_run-resized-gray.log


# Custom dataset links:
http://groups.csail.mit.edu/vision/SUN/
Brackish
Project Natick
https://www.researchgate.net/post/Can_anybody_help_me_find_some_underwater_image_or_video_data_set3 # this has many links


### Preparing Yolo Darknet files:
# Color B1: synthetic
python darknet_dataset_creator.py --dataset-name synthetic --data-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic --classes mine --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic

# Color M1: synthetic-NST
python darknet_dataset_creator.py --dataset-name synthetic-NST --data-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-NST --classes mine --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST

# Gray B2: synthetic-gray
python darknet_dataset_creator.py --dataset-name synthetic-gray --data-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray --classes mine --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray

# Gray M2.1: synthetic-gray-NST
python darknet_dataset_creator.py --dataset-name synthetic-gray-NST --data-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST --classes mine --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST

# Gray M2.1: synthetic-NST-gray
python darknet_dataset_creator.py --dataset-name synthetic-NST-gray --data-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-NST-gray --classes mine --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray


### Running Yolo training.
# Image sizes in all the above datasets.
num_images: 984, train: 788, val: 196
height: 256, width: 192
batch_size: 64
steps_per_epoch: 13
update classes and filters.

# Run from code directory.
# Train Color M1: synthetic
/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/synthetic.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74 &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/yolo_training.log


# Color M1: synthetic-NST
/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/synthetic-NST.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74 &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/yolo_training.log

# Gray B2: synthetic-gray
/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/synthetic-gray.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74 &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/yolo_training.log

# Gray M2.1: synthetic-gray-NST
/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/synthetic-gray-NST.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74 &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/yolo_training.log

# Gray M2.1: synthetic-NST-gray
/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/synthetic-NST-gray.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74 &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/yolo_training.log


# Check the average loss in the last epoch:
less /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/yolo_training.log | tail -1

less /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/yolo_training.log | tail -1

less /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/yolo_training.log | tail -1

less /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/yolo_training.log | tail -1

less /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/yolo_training.log | tail -1


### Evaluating trained Yolo models.
# Run from darknet folder.

### Test running detection.
./darknet detector test /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/synthetic.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/darknet_backup/yolov3-wells_final.weights -dont_show -ext_output -save_labels -thresh 0.01 adam-eperjesi-et7JPPrMtIw-unsplash-401.jpg

### Ground Truth Data Organization:
# Resize test images.
from PIL import Image
import glob
import os
import shutil

def resize_retain_aspect_ratio(src_dir, dest_dir, max_x_shape=256, max_y_shape=256, img_ext="png"):
    os.makedirs(dest_dir)
    size = max_x_shape, max_y_shape
    input_images = glob.glob(os.path.join(src_dir, "*.%s" % img_ext))
    for infile in input_images:
        im = Image.open(infile)
        im.thumbnail(size, Image.ANTIALIAS)
        outfile = os.path.join(dest_dir, os.path.basename(infile))
        im.save(outfile, img_ext)

    print("The images have been resized in the dest_dir: %s" % dest_dir)


resize_retain_aspect_ratio("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mitchell_water_mine/darknet/mine_images_orig",
                           "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mitchell_water_mine/darknet/mine_images",
                           max_x_shape=256, max_y_shape=256, img_ext="png")


# Convert label format to VOC - [<class_name> <left> <top> <right> <bottom> [<difficult>]] format, from Yolo format - [center_x, center_y, width, height] (relative to image width and height).

from skimage import io

import glob
import os

def convert_yolo_to_voc_bbox_format(src_dir, dest_dir, src_images_dir, img_ext="png", class_map=None, add_one=True):
    def _convert_line(yolo_format_line, image_file):
        img = io.imread(image_file)
        img_w = img.shape[1]
        img_h = img.shape[0]

        toks = yolo_format_line.split(" ")
        class_name = toks[0]
        if class_map:
            class_name = class_map[class_name]

        b_x = float(toks[1])
        b_y = float(toks[2])
        b_w = float(toks[3])
        b_h = float(toks[4])

        # Refer to conversion code from 
        # - https://github.com/Cartucho/mAP/blob/master/scripts/extra/convert_gt_yolo.py#L15
        # - https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        left  = int((b_x-b_w/2.) * img_w)
        right = int((b_x+b_w/2.) * img_w)
        top   = int((b_y-b_h/2.) * img_h)
        bot   = int((b_y+b_h/2.) * img_h)

        if left < 0:
            left = 0
        if right > img_w-1:
            right = img_w-1
        if top < 0:
            top = 0
        if bot > img_h-1:
            bot = img_h-1

        # In the official VOC challenge the top-left pixel in the image has coordinates (1;1)
        if add_one:
            left += 1
            right += 1
            top +=1
            bot += 1

        voc_format_line = " ".join(map(str, [class_name, left, top, right, bot]))
        print("New voc_format_line is: %s" % voc_format_line)
        return voc_format_line

    os.makedirs(dest_dir)

    input_labels = glob.glob(os.path.join(src_dir, "*.txt"))
    for infile in input_labels:
        outfile = os.path.join(dest_dir, os.path.basename(infile))
        with open(outfile, "w") as out_fp:
            with open(infile) as in_fp:
                for line in in_fp:
                    cur_img_name = "%s.%s" % (os.path.splitext(os.path.basename(infile))[0], img_ext)
                    cur_img_file = os.path.join(src_images_dir, cur_img_name)
                    if not os.path.exists(cur_img_file):
                        raise ValueError("The correct image not found: %s" % cur_img_file)

                    converted_line = _convert_line(line.strip(), cur_img_file)
                    out_fp.write("%s\n" % converted_line)

    print("The labels have been converted in the dest_dir: %s" % dest_dir)


convert_yolo_to_voc_bbox_format("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mitchell_water_mine/darknet/mine_labels_orig",
                                "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mitchell_water_mine/darknet/mine_labels",
                                "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mitchell_water_mine/darknet/mine_images",
                                img_ext="png",
                                class_map={"0": "0"},  # {"0": "mine"} previous run.
                                add_one=True)

### Testing and Evaluating Trained Yolo Models.
# Get the Yolo detections: run_yolo_detector.py.

# Test the script: run_yolo_detector.py
python run_yolo_detector.py --input-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/input_images --yolo-config-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg --yolo-data-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/synthetic-NST.data --yolo-weights-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/darknet_backup/yolov3-wells_final.weights --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/detector_output --image-ext jpg

# Get mAP evaluation script:
https://github.com/Cartucho/mAP

# Test evaluation script.
convert_yolo_to_voc_bbox_format("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/detector_output/ground_truth_files-yolo",
                                "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/detector_output/ground_truth_files-voc",
                                "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/detector_output/test_images_detected_labels",
                                img_ext="jpg",
                                class_map={"0": "0"},
                                add_one=True)

python evaluate_detections.py --gt-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/detector_output/ground_truth_files-voc --dr-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/detector_output/preds_voc_format --images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/detector_output/test_images_detected_labels --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/test_detector/detector_output/evaluation --image-ext jpg


# Run yolo detection and evaluations on mine test set from all 5 models.
# Color M1: synthetic
python run_yolo_detector.py --input-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images --yolo-config-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg --yolo-data-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/synthetic.data --yolo-weights-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/darknet_backup/yolov3-wells_final.weights --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/mine_test_data-mono_detector_output --image-ext png &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/mine_test_data-mono_detector_output.log

python evaluate_detections.py --gt-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_labels --dr-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/mine_test_data-mono_detector_output/preds_voc_format --images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/mine_test_data-mono_detector_output/test_images_detected_labels --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/mine_test_data-mono_detector_output/evaluation --image-ext png


# Color M1: synthetic-NST
python run_yolo_detector.py --input-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images --yolo-config-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg --yolo-data-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/synthetic-NST.data --yolo-weights-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/darknet_backup/yolov3-wells_final.weights --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/mine_test_data-mono_detector_output --image-ext png &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/mine_test_data-mono_detector_output.log

python evaluate_detections.py --gt-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_labels --dr-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/mine_test_data-mono_detector_output/preds_voc_format --images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/mine_test_data-mono_detector_output/test_images_detected_labels --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST/mine_test_data-mono_detector_output/evaluation --image-ext png


# Gray B2: synthetic-gray
python run_yolo_detector.py --input-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images --yolo-config-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg --yolo-data-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/synthetic-gray.data --yolo-weights-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/darknet_backup/yolov3-wells_final.weights --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output --image-ext png &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output.log

python evaluate_detections.py --gt-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_labels --dr-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output/preds_voc_format --images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output/test_images_detected_labels --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output/evaluation --image-ext png


# Gray M2.1: synthetic-gray-NST
python run_yolo_detector.py --input-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images --yolo-config-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg --yolo-data-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/synthetic-gray-NST.data --yolo-weights-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/darknet_backup/yolov3-wells_final.weights --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/mine_test_data-mono_detector_output --image-ext png &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/mine_test_data-mono_detector_output.log

python evaluate_detections.py --gt-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_labels --dr-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/mine_test_data-mono_detector_output/preds_voc_format --images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/mine_test_data-mono_detector_output/test_images_detected_labels --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray-NST/mine_test_data-mono_detector_output/evaluation --image-ext png


# Gray M2.1: synthetic-NST-gray
python run_yolo_detector.py --input-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images --yolo-config-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg --yolo-data-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/synthetic-NST-gray.data --yolo-weights-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/darknet_backup/yolov3-wells_final.weights --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/mine_test_data-mono_detector_output --image-ext png &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/mine_test_data-mono_detector_output.log

python evaluate_detections.py --gt-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_labels --dr-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/mine_test_data-mono_detector_output/preds_voc_format --images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/mine_test_data-mono_detector_output/test_images_detected_labels --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-NST-gray/mine_test_data-mono_detector_output/evaluation --image-ext png


### Next questions to address:
# How does adding x/2, x amount of in-situ background images into the pipeline help in improving the performance?
# How does resizing the image to 512 instead 256 affect the performance?
# How does their performance compare on color test images? [can we do this for Mines? Maybe I can collect some data if needed, as it allows us to compare the same models, making it easy to compare and learn]
# Maybe write a paper or poster on it.

# ---------------------------------------------------
### Automating the pipeline.

# Script to run end to end model, where background images are input, along with the pretrained weights to be considered for Yolo, and an output directory where all the results would be written.
# Steps of this pipeline.
- Step 1:
# Create Synthetic data by rendering the object on the background images: input orignal size images, and the .egg file for object 3d model, and number fof scenes need (have some margin which we might have to ignore).
python run_yolo_data_renderer.py --background-dir ../data/underwater_background/unsplash/unsplash_underwater_collection --object-model-file ../panda3d_models/mine.egg --num-scenes 1000 --out-dir ../data/darknet_datasets/unsplash_mine_raw/darknet_images_labels --max-objects 2

<resize the images rendered, but the same labels would be fine>

- Step 2:
# Resize the images to 256, and convert grayscale version copy of the rendered data.
# Convert the input background images also to grayscale as a copy.

- Step 3
# Run NST on 2 datasets (on color and on grayscale). Same command shown for grayscale.
python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection-gray --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST --max-dim 256 &> ../logs/nst_run-resized-gray.log

- Step 4:
# Create darknet datasets format: 5 models
python darknet_dataset_creator.py --dataset-name synthetic-gray --data-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray --classes mine --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray

- Step 5:
# Run darknet yolo training: 5 models
/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/synthetic-gray.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74 &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/yolo_training.log

<log the validation loss: less /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/yolo_training.log | tail -1>

<Delete the weights files except the final.weights.>

#-----
# Testing: Assuming that the testbed data is organized correctly.
# It requires resizing, and voc-format labels instead of yolo format labels.
- Step 6:
# Run detections on the testbed: 5 models.
python run_yolo_detector.py --input-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images --yolo-config-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg --yolo-data-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/synthetic-gray.data --yolo-weights-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/darknet_backup/yolov3-wells_final.weights --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output --image-ext png &> /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output.log

- Step 7:
# Run evaluation on the testbed: 5 models.
python evaluate_detections.py --gt-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_labels --dr-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output/preds_voc_format --images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output/test_images_detected_labels --output-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic-gray/mine_test_data-mono_detector_output/evaluation --image-ext png

# Script for running the above 7 steps.


### Preparing insitu data consisting 250 background images, then consider the rest 105 background images in the test data.

# Converted all png images to jpg images: 'mogrify -format jpg *.png && rm *.png'

import glob
import os
import random
import shutil

src_background_images = glob.glob("/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images_orig/bg_*.jpg")
dest_train_background_images = "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/insitu-250-gray"

num_train = 250
random.shuffle(src_background_images)
selected_train_background_images = src_background_images[:num_train]

for image in selected_train_background_images:
    shutil.copy(image, os.path.join(dest_train_background_images, os.path.basename(image)))


# Remove the selected images from the original test set.
test_data_dirs = [
    "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images",
    "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_images_orig",
    "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_labels",
    "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/mine_labels_orig"
]

selected_basenames = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], selected_train_background_images))


for test_data_dir in test_data_dirs:
    for selected_basename in selected_basenames:
        cur_files = glob.glob(os.path.join(test_data_dir, "%s.*" % selected_basename))
        assert len(cur_files) == 1
        cur_file = cur_files[0]
        os.remove(cur_file)

# Create subsets of the train background images.
train_background_images_dir = "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/insitu-250-gray"
train_background_images = glob.glob(os.path.join(train_background_images_dir, "*.jpg"))

dest_path_prefix = "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/insitu-"
num_train_data_subsets = [50, 100, 150, 200]

for num_train_data_subset in num_train_data_subsets:
    cur_dest_dir = dest_path_prefix + "%s-gray" % num_train_data_subset
    os.makedirs(cur_dest_dir)
    random.shuffle(train_background_images)
    cur_selected_images = train_background_images[:num_train_data_subset]
    for cur_image in cur_selected_images:
        shutil.copy(cur_image, os.path.join(cur_dest_dir, os.path.basename(cur_image)))


# Test set path: /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set
# Train background images: /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/insitu-<>-gray  # 50, 100, 150, 200, 250

### Run the Automated Pipeline.
# Test sample.
python run_synthetic_data_training_and_evaluation.py --background-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/insitu/insitu-50-gray --is-grayscale --object-model-file /home/bhuvan/Projects/underwater_synthetic_image_recognition/panda3d_models/mine.egg --classes mine --num-train-scenes 4 --max-objects 2 --image-size 256 --image-ext jpg --previous-data '{"synthetic": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-sample", "nst": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST-sample"}' --gt-images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_images --gt-labels-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_labels --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-sample

# Run the end-to-end pipeline for 50, 100, 150, 200, 250 insitu images.
# Include the images from previous network, and start from scratch instead of pre-trained weights from synthetic backgrounds then train for full 15000 steps.
python run_synthetic_data_training_and_evaluation.py --background-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/insitu/insitu-50-gray --is-grayscale --object-model-file /home/bhuvan/Projects/underwater_synthetic_image_recognition/panda3d_models/mine.egg --classes mine --num-train-scenes 200 --max-objects 2 --previous-data '{"synthetic": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray", "nst": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST"}' --image-size 256 --image-ext jpg --gt-images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_images --gt-labels-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_labels --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray > /home/bhuvan/Projects/underwater_synthetic_image_recognition/logs/analysis_insitu-50-gray.log 2>&1


python run_synthetic_data_training_and_evaluation.py --background-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/insitu/insitu-100-gray --is-grayscale --object-model-file /home/bhuvan/Projects/underwater_synthetic_image_recognition/panda3d_models/mine.egg --classes mine --num-train-scenes 400 --max-objects 2 --previous-data '{"synthetic": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray", "nst": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST"}' --image-size 256 --image-ext jpg --gt-images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_images --gt-labels-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_labels --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-100-gray > /home/bhuvan/Projects/underwater_synthetic_image_recognition/logs/analysis_insitu-100-gray.log 2>&1


python run_synthetic_data_training_and_evaluation.py --background-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/insitu/insitu-150-gray --is-grayscale --object-model-file /home/bhuvan/Projects/underwater_synthetic_image_recognition/panda3d_models/mine.egg --classes mine --num-train-scenes 600 --max-objects 2 --previous-data '{"synthetic": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray", "nst": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST"}' --image-size 256 --image-ext jpg --gt-images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_images --gt-labels-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_labels --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-150-gray > /home/bhuvan/Projects/underwater_synthetic_image_recognition/logs/analysis_insitu-150-gray.log 2>&1


python run_synthetic_data_training_and_evaluation.py --background-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/insitu/insitu-200-gray --is-grayscale --object-model-file /home/bhuvan/Projects/underwater_synthetic_image_recognition/panda3d_models/mine.egg --classes mine --num-train-scenes 800 --max-objects 2 --previous-data '{"synthetic": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray", "nst": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST"}' --image-size 256 --image-ext jpg --gt-images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_images --gt-labels-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_labels --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-200-gray > /home/bhuvan/Projects/underwater_synthetic_image_recognition/logs/analysis_insitu-200-gray.log 2>&1


python run_synthetic_data_training_and_evaluation.py --background-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/insitu/insitu-250-gray --is-grayscale --object-model-file /home/bhuvan/Projects/underwater_synthetic_image_recognition/panda3d_models/mine.egg --classes mine --num-train-scenes 1000 --max-objects 2 --previous-data '{"synthetic": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray", "nst": "/home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/synthetic-gray-NST"}' --image-size 256 --image-ext jpg --gt-images-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_images --gt-labels-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/water_mine/mine_test_data-mono/darknet/reduced_bg_test_set/mine_labels --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-250-gray > /home/bhuvan/Projects/underwater_synthetic_image_recognition/logs/analysis_insitu-250-gray.log 2>&1 [COMPLETED] # Dir not perform well. Might have to retrain and check.

# Failed due to improper renders. Also found that for insite-50 (for which it completed), it gave very less AUC: 34%.
# [TODO]Infer again on test set with those zoomed-in mines removed.

# Add real-mine images to train data.
# realmine_reduced_test_set
10 images with real-mines have been sampled out from the test-set.
They have copied 20 times, so that there are 200 such images. This is done so that it can be compared with the insitu-50, where 200 scenes with insitu background are included in the data.
These 200 real mine images have been put to both synthetic-gray and synthetic-gray-NST rendered images copied from analysis_insitu-50-gray to the new dir: analysis_insitu-50-gray-realmine-10.

# Train Yolo models on analysis_insitu-50-gray-realmine-10, synthetic-gray and dynthetic-gray-NST.
# Step 4: Creating Yolo training files as required by Darknet.
python darknet_dataset_creator.py --dataset-name synthetic-gray --classes mine --data-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/rendered_images/synthetic-gray --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray > /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/logs/darknet_dataset_creator-synthetic.log 2>&1

python darknet_dataset_creator.py --dataset-name synthetic-gray-NST --classes mine --data-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/rendered_images/synthetic-gray-NST --out-dir /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray-NST > /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/logs/darknet_dataset_creator-nst.log 2>&1

# Step 5: Run Yolo training on Darknet.
/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray/synthetic-gray.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74 > /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray/yolo_training.log 2>&1 [IN PROCESS]

/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray-NST/synthetic-gray-NST.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74 > /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray-NST/yolo_training.log 2>&1 [IN PROCESS]


TODO: Check the validation loss of the trained yolo model (synthetic) from the logs: /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray/yolo_training.log
TODO: Deleting the intermediate weight files to save disk memory from the backup dir: /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray/darknet_backup

TODO: Check the validation loss of the trained yolo model (nst) from the logs: /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray-NST/yolo_training.log
TODO: Deleting the intermediate weight files to save disk memory from the backup dir: /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/analysis_insitu-50-gray-realmine-10/yolo_training_files/synthetic-gray-NST/darknet_backup



# Test Set Mine_images dir-size
full_orig_test_set: 610 - 1
reduced_bg_test_set: 360 - 1
realmine_reduced_test_set: 350 - 1
realmine_reduced_test_set_simplified: TODO
