# Commands.

# Docker:
docker run --gpus all -itd -p 8888:8888 -p 16006:16006 tensorflow/tensorflow:latest-gpu-py3 bash

docker exec -ti a0bb7eeed020 bash

docker stop a0bb7eeed020 ; docker rm a0bb7eeed020

jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

docker cp path_in_local a0bb7eeed020:/projects/

# All scripts run from the code directory.

# Yolo data synthetic renderer.
python yolo_data_renderer.py --background-dir ../data/underwater_background/unsplash/unsplash_underwater_collection --object-model-file ../panda3d_models/Torus.egg --num-scenes 1000 --out-dir ../data/darknet_datasets/unsplash_rendered_test/darknet_images_labels --max-objects 2 --custom-start-index -2

# Darknet

### Data preparation:
python darknet_dataset_creator.py --dataset-name unsplash_rendered_test --data-dir ../data/darknet_datasets/unsplash_rendered_test/darknet_images_labels --classes torus --out-dir ../data/darknet_datasets/unsplash_rendered_test

### Training.
./darknet detector train /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_rendered_test/unsplash_rendered_test.data /home/bhuvan/Projects/underwater_synthetic_image_recognition/data/darknet_datasets/unsplash_rendered_test/yolov3-tiny.cfg darknet53.conv.74
