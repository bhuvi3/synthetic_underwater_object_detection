#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:14:00 2019

@author: mitchell
"""

from collections import defaultdict

import cv2
import glob
import matplotlib.pyplot as plt
import os
import sys

try:
    images_labels_dir_name = sys.argv[1]
except:
    print("Provide the path to directory containing images and labels as the first command line argument.")
    raise

image_files = glob.glob(os.path.join(images_labels_dir_name, "*.jpg"))
image_files = sorted(image_files)

image_shape_dict = defaultdict(int)
for img_path in image_files:
    img = cv2.imread(img_path)
    img_width = img.shape[1]
    img_height = img.shape[0]

    label = str(img_path.replace('jpg', 'txt'))

    print("Image-Label: %s | %s" % (os.path.basename(img_path), os.path.basename(label)))
    with open(label.rstrip()) as label:
        for _bbox in label:
            bbox = _bbox.split(' ')
            x_min = float(bbox[1])
            y_min = float(bbox[2])
            box_width = float(bbox[3])
            box_height = float(bbox[4])
            x = int(float(x_min - box_width/2)*img_width)
            y = int(float(y_min - box_height/2)*img_height)
            left_top = (x, y)
            x2 = int(float(x_min + box_width/2)*img_width)
            y2 = int(float(y_min + box_height/2)*img_height)
            bottom_right = (x2, y2)

            center = (int(x_min*img_width), int(y_min*img_height))
            cv2.circle(img, left_top, 10, (255,0,255))
            cv2.circle(img, center, 10, (0,255,0))
            cv2.rectangle(img, left_top, bottom_right, (255,0,0), 2)

    image_shape_dict[img.shape] += 1

    if (img.shape[0] > 60):
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(im_rgb)
        plt.show()
        plt.clf()

print(image_shape_dict)
