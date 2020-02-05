#!/usr/bin/env python3

"""
Runs the Darknet Yolo detector on a trained model and outputs the detection in Yolo and VOC format.

"""

from skimage import io

import argparse
import glob
import os
import re
import shutil
import subprocess

DARKNET_DIR_PATH = "/home/bhuvan/Projects/darknet_AlexeyAB"


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


def _convert_pred_line(yolo_format_line, image_file, class_map=None, add_one=True):
    img = io.imread(image_file)
    img_w = img.shape[1]
    img_h = img.shape[0]

    toks = yolo_format_line.split(" ")
    class_name = toks[0]
    if class_map:
        class_name = class_map[class_name]

    # In case of ground truth lines, the confidence score can be skipped.
    confidence = toks[1]

    b_x = float(toks[2])
    b_y = float(toks[3])
    b_w = float(toks[4])
    b_h = float(toks[5])

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

    voc_format_line = " ".join(map(str, [class_name, confidence, left, top, right, bot]))
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
    def _get_pred_lines(raw_output_file, pred_label_file):
        pred_dicts = []
        with open(pred_label_file) as fp:
            for line in fp:
                toks = line.strip().split(" ")
                cur_pred_dict = {
                    "class": toks[0],
                    "yolo_coord_1": toks[1],
                    "yolo_coord_2": toks[2],
                    "yolo_coord_3": toks[3],
                    "yolo_coord_4": toks[4]
                }
                pred_dicts.append(cur_pred_dict)

        num_preds = len(pred_dicts)

        # Parse the raw output to get the corresponding confidence scores.
        # The predictions are printed after the line starting with the name of the input test image.
        confidence_scores = []
        with open(raw_output_file) as fp:
            flag = False
            for line in fp:
                if not flag:
                    if line.strip().startswith("%s: Predicted" % os.path.basename(test_image_path)):
                        flag = True
                else:
                    cur_confidence_score = str(float(re.split(r"\s+", line.strip())[1].strip("%")) / 100)
                    confidence_scores.append(cur_confidence_score)

        if len(confidence_scores) != num_preds:
            raise ValueError("The number of predicted lines in ext_output and saved_labels do not match: (%s, %s)"
                             % (len(confidence_scores), num_preds))

        # Include confidence score with the pred_dict and format the pred_line.
        pred_lines = []
        for cur_pred_dict, cur_confidence_score in zip(pred_dicts, confidence_scores):
            cur_pred_line = " ".join([
                cur_pred_dict["class"],
                cur_confidence_score,
                cur_pred_dict["yolo_coord_1"],
                cur_pred_dict["yolo_coord_2"],
                cur_pred_dict["yolo_coord_3"],
                cur_pred_dict["yolo_coord_4"],
            ])
            pred_lines.append(cur_pred_line)

        return pred_lines

    # Run darknet detector test with thresh 0.001 (or 0)? Run in a subprocess, redirect output to raw_output_file.
    os.chdir(DARKNET_DIR_PATH)
    cmd = [
        "./darknet", "detector", "test", "-dont_show", "-ext_output", "-save_labels",
        "-thresh", "0.01",
        yolo_data_path,
        yolo_config_path,
        yolo_weights_path,
        test_image_path
    ]
    with open(raw_output_file, "w") as raw_out_fp:
        print("\nRunning command: %s" % " ".join(cmd))
        error_status = subprocess.run(cmd, stdout=raw_out_fp, stderr=raw_out_fp)
        print("Command return the error status: %s" % error_status)

    # The 'darknet' script writes the predicted label in Yolo format in the same directory of the input image.
    pred_label_file = "%s.txt" % os.path.splitext(test_image_path)[0]

    # Copy the predicted marked image from current directory to the output directory, as the 'darknet' command generates
    # the marked image in the current directory.
    # Note that 'darknet' always creates the marked images in 'jpg' format.
    shutil.move("./predictions.jpg", marked_image_out_file)

    # Get raw pred lines, by parsing the output log in previous script.
    yolo_pred_lines = _get_pred_lines(raw_output_file, pred_label_file)
    return yolo_pred_lines


def run_yolo_detections(input_dir, yolo_config_path, yolo_data_path, yolo_weights_path, output_dir, image_ext):
    out_marked_images_dir = os.path.join(output_dir, "marked_images")
    out_yolo_format_preds_dir = os.path.join(output_dir, "preds_yolo_format")
    out_voc_format_preds_dir = os.path.join(output_dir, "preds_voc_format")
    raw_output_dir = os.path.join(output_dir, "raw_output")

    for dir in [output_dir, out_marked_images_dir, out_yolo_format_preds_dir, out_voc_format_preds_dir, raw_output_dir]:
        os.makedirs(dir)

    # Copy the test images inside the output directory, inside which the darknet saves the detected labels.
    out_test_images_dir = os.path.join(output_dir, "test_images_detected_labels")
    shutil.copytree(input_dir, out_test_images_dir)

    print("Started Yolo Detections.")
    input_image_files = glob.glob(os.path.join(out_test_images_dir, "*.%s" % image_ext))
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
        # Note that 'darknet' always creates the marked images in 'jpg' format.
        marked_image_out_file = os.path.join(out_marked_images_dir,
                                             "%s.jpg" % os.path.splitext(os.path.basename(input_image))[0])

        yolo_pred_lines = run_darknet_detector(input_image,
                                               yolo_config_path,
                                               yolo_data_path,
                                               yolo_weights_path,
                                               raw_output_file,
                                               marked_image_out_file)

        voc_pred_lines = [_convert_pred_line(x, input_image, add_one=True) for x in yolo_pred_lines]

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
