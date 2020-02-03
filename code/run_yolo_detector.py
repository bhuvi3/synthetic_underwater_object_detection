#!/usr/bin/env python3

"""
Runs the Darknet Yolo detector on a trained model and outputs the detection in yolo and voc format.

"""

from processify import processify
from skimage import io

import argparse
import glob
import os
