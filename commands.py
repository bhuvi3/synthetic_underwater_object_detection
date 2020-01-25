# Commands.

# Docker:
docker run --gpus all -itd -p 8888:8888 -p 16006:16006 tensorflow/tensorflow:latest-gpu-py3 bash

docker exec -ti a0bb7eeed020 bash

docker stop a0bb7eeed020 ; docker rm a0bb7eeed020

jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

docker cp path_in_local a0bb7eeed020:/projects/

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

# Running NST on resized images.
python neural_style_transfer.py --content-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels --style-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/underwater_background/unsplash/unsplash_underwater_collection --out-path /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_mine_raw/darknet_images_labels-NST --grayscale --max-dim 256 &> ../logs/nst_run-resized.log
