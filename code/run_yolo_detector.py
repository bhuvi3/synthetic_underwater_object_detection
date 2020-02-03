#!/usr/bin/env python3

"""
Runs the Darknet Yolo detector on a trained model and outputs the detection in Yolo and VOC format.

"""

from skimage import io

import argparse
import glob
import os

DARKNET_SCRIPT_PATH = "/home/bhuvan/Projects/darknet_AlexeyAB/darknet"


def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Runs the Darknet Yolo detector on a trained model and outputs the detection in Yolo and VOC format.
    """)
    parser.add_argument('--input-dir',
                        required=True,
                        help="The path to the input directory containing the test images.")
    parser.add_argument('--yolo-config-path',
                        required=True,
                        help="The path to the Yolo config file for Darknet.")
    parser.add_argument('--yolo-data-path',
                        required=True,
                        help="The path to the Yolo data file for Darknet.")
    parser.add_argument('--yolo-weights-path',
                        required=True,
                        help="The path to the Yolo weights file for Darknet.")
    parser.add_argument('--output-dir',
                        required=True,
                        help="The path to the output directory to which the Yolo detection outputs will be written.")
    parser.add_argument('--image-ext',
                        default="jpg",
                        help="The extension of the image files in the --data-dir. Default: 'jpg'.")

    args = parser.parse_args()
    return args


def _convert_line(yolo_format_line, image_file, class_map=None, add_one=True):
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
    left = int((b_x - b_w / 2.) * img_w)
    right = int((b_x + b_w / 2.) * img_w)
    top = int((b_y - b_h / 2.) * img_h)
    bot = int((b_y + b_h / 2.) * img_h)

    if left < 0:
        left = 0
    if right > img_w - 1:
        right = img_w - 1
    if top < 0:
        top = 0
    if bot > img_h - 1:
        bot = img_h - 1

    # In the official VOC challenge the top-left pixel in the image has coordinates (1;1)
    if add_one:
        left += 1
        right += 1
        top += 1
        bot += 1

    voc_format_line = " ".join(map(str, [class_name, left, top, right, bot]))
    print("New voc_format_line is: %s" % voc_format_line)
    return voc_format_line


def run_darknet_detector(test_image_path,
                         yolo_config_path,
                         yolo_data_path,
                         yolo_weights_path,
                         raw_output_file,
                         marked_image_out_file):
    """
    Runs the Darknet detector in 'test' mode on a single image (with thresh 0.001) and gets the detection output lines.
    It runs on a subprocess.

    """
    # Run darknet detector test with thresh 0.001 (or 0)? Run in a subprocess, redirect output to raw_output_file.
    # Get raw pred lines, by parsing the output log in previous script.
    # Correct the coords to Yolo relative format.
    # Move the marked image to marked_image_out_file
    pass


def run_yolo_detections(input_dir, yolo_config_path, yolo_data_path, yolo_weights_path, output_dir, image_ext):
    out_marked_images_dir = os.path.join(output_dir, "marked_images")
    out_yolo_format_preds_dir = os.path.join(output_dir, "preds_yolo_format")
    out_voc_format_preds_dir = os.path.join(output_dir, "preds_voc_format")
    raw_output_dir = os.path.join(output_dir, "raw_output")
    for dir in [output_dir, out_marked_images_dir, out_yolo_format_preds_dir, out_voc_format_preds_dir, raw_output_dir]:
        os.makedirs(dir)

    input_image_files = glob.glob(os.path.join(input_dir, "*.%s" % image_ext))
    num_test_images = len(input_image_files)
    count = 0
    for input_image in input_image_files:
        count += 1
        if count % 100 == 0:
            print("Processed %s/%s test images." % (count, num_test_images))

        raw_output_file = os.path.join(raw_output_dir, "%s.log" % os.path.splitext(os.path.basename(input_image))[0])
        yolo_pred_file = os.path.join(out_yolo_format_preds_dir,
                                      "%s.txt" % os.path.splitext(os.path.basename(input_image))[0])
        voc_pred_file  = os.path.join(out_voc_format_preds_dir,
                                      "%s.txt" % os.path.splitext(os.path.basename(input_image))[0])
        marked_image_out_file = os.path.join(out_marked_images_dir,
                                             "%s.%s" % (os.path.splitext(os.path.basename(input_image))[0], image_ext))

        yolo_pred_lines = run_darknet_detector(input_image,
                                               yolo_config_path,
                                               yolo_data_path,
                                               yolo_weights_path,
                                               raw_output_file,
                                               marked_image_out_file)

        voc_pred_lines = [_convert_line(x, input_image, add_one=True) for x in yolo_pred_lines]

        for cur_pred_file, cur_pred_lines in [(yolo_pred_file, yolo_pred_lines), (voc_pred_file, voc_pred_lines)]:
            with open(cur_pred_file, "w") as fp:
                for line in cur_pred_lines:
                    fp.write("%s\n" % line)

    print("Processed %s test images, and the outputs have been written to %s" % (num_test_images, output_dir))


if __name__ == "__main__":
    args = get_args()
    run_yolo_detections(args.input_dir,
                        args.yolo_config_path,
                        args.yolo_data_path,
                        args.yolo_weights_path,
                        args.output_dir,
                        args.image_ext)
