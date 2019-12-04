#!/usr/bin/python

import argparse
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Get yolo dataset from background images and the 3d model rendered into a non-photo realistic synthetic scene.
    """)
    parser.add_argument('--background-dir',
                        required=True,
                        help="The path to the directory containing the background images.")
    parser.add_argument('--object-model-file',
                        required=True,
                        help="The path to the Panda3d object 3d-model file (.egg).")
    parser.add_argument('--num-scenes',
                        type=int,
                        required=True,
                        help="The number of scenes to generate.")
    parser.add_argument('--out-dir',
                        required=True,
                        help="The path to the output dir to which the yolo dataset need to be written.")
    parser.add_argument('--max-objects',
                        type=int,
                        default=2,
                        help="The maximum number of objects to be rendered on a scene.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    for cur_count in range(args.num_scenes):
        cur_count += 1
        print("Processing scene: %s (out of %s)" % (cur_count, args.num_scenes))
        command = ["python", "/home/bhuvan/Projects/underwater_synthetic_image_recognition/code/yolo_data_renderer.py",
                   "--background-dir", args.background_dir,
                   "--object-model-file", args.object_model_file,
                   "--num-scenes", "1",
                   "--out-dir", args.out_dir,
                   "--max-objects", str(args.max_objects),
                   "--count-offset", str(cur_count)]
        print("Running command: %s" % " ".join(command))
        error_status = subprocess.run(command)
        print("Error status: %s" % error_status)
