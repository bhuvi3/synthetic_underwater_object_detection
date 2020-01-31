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
/home/bhuvan/Projects/darknet/darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/yolo_training_files/synthetic/synthetic.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/yolov3-wells.cfg /home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_cfg/darknet53.conv.74


### Evaluating trained Yolo models.
