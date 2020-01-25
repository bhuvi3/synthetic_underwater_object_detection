#!/usr/bin/python

"""
The script performs neural style transfer on given pair of style image and content image to create a transformed image.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import PIL.Image

import argparse
import glob
import os
import shutil
import time

# XXX:Restrict GPU for TF, since it is not able to access CUDA operations.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Performs neural style transfer on given pair of style image and content image to create a transformed image.
    """)
    parser.add_argument('--content-path',
                        required=True,
                        help="The path (or dir) to the content image. "
                             "If directory, the folder must contain the annotation file as well.")
    parser.add_argument('--style-path',
                        required=True,
                        help="The path (or dir) to the style image.")
    parser.add_argument('--out-path',
                        required=True,
                        help="The path to the output file (or dir) to which the transformed image needs to be written.")
    parser.add_argument('--weights',
                        default="imagenet",
                        help="Indicate the weights for the VGG network to retrieve the style and the content layers.")
    parser.add_argument('--grayscale',
                        action="store_true",
                        help="Boolean flag to indicate if the grayscale version of the transformed image needs"
                             "to be saved. In single mode, creates a new file, and in multiple-mode, "
                             "it creates a new directory.")
    parser.add_argument('--max-dim',
                        type=int,
                        default=512,
                        help="The maximum dimension to which the images are resized "
                             "before applying Neural Style Transfer.")

    # TODO: Add more options for separate pretrained weights for style and content images.
    args = parser.parse_args()
    return args


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, max_dim):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def vgg_layers(layer_names, weights):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights=weights)
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers, weights):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers, weights)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def apply_neural_style_transfer(content_path, style_path, max_dim, weights='imagenet'):
    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    # Load the Content and Style images.
    content_image = load_img(content_path, max_dim)
    style_image = load_img(style_path, max_dim)

    # Define content and style representations.
    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer of interest
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    extractor = StyleContentModel(style_layers, content_layers, weights)

    # Run Gradient Descent.
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # Start training.
    start = time.time()

    style_weight = 1e-2
    content_weight = 1e4
    total_variation_weight = 30
    epochs = 10
    steps_per_epoch = 100

    image = tf.Variable(content_image)
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)

    transformed_image = tensor_to_image(image)

    end = time.time()
    print("Total time: {:.1f}".format(end - start))
    return transformed_image


def process_neural_style_transfer(args):
    if not os.path.isdir(args.content_path):
        print("Processing Neural Style Transfer for single image.")
        assert os.path.isdir(args.style_path) is False
        transformed_image = apply_neural_style_transfer(args.content_path,
                                                        args.style_path,
                                                        args.max_dim,
                                                        weights=args.weights)
        transformed_image.save(args.out_path)
        if args.grayscale:
            transformed_image.convert('L').save("%s-gray%s" % os.path.splitext(args.out_path))
        print("Processed the Neural Style Transfer and output written to %s." % args.out_path)
        return

    # Assumes that style and content paths are directory containing multiple files.
    print("Processing Neural Style Transfer for JPG images in the directory (content-path).")
    assert os.path.isdir(args.style_path) is True
    os.makedirs(args.out_path)
    if args.grayscale:
        gray_out_dir = "%s-gray" % args.out_path
        os.makedirs(gray_out_dir)
    print("Applying Neural Style transfer to content images (JPG) in the content-path, and referring to "
          "matching style image names in the style-path. Writing the style transferred images to out-path. "
          "If annotation files are found in the content-path, they will copied to the out-path as well")

    content_image_paths = sorted(glob.glob(os.path.join(args.content_path, "*.jpg")))
    num_content_images = len(content_image_paths)
    c = 0
    for cur_content_path in content_image_paths:
        c += 1
        try:
            print("Processing NST for content image: %s (out of %s)" % (c, num_content_images))
            cur_content_file_basename = os.path.splitext(os.path.basename(cur_content_path))[0]
            cur_style_path = os.path.join(args.style_path, "%s.jpg" % "-".join(cur_content_file_basename.split("-")[:-1]))
            assert os.path.exists(cur_style_path), "Style image %s was not found." % cur_style_path
            cur_out_path = os.path.join(args.out_path, "%s.jpg" % cur_content_file_basename)
            cur_anno_path = os.path.join(args.content_path, "%s.txt" % cur_content_file_basename)
            if not os.path.exists(cur_anno_path):
                print("Annotation file doesn't exist: %s. Skipping the content file." % cur_anno_path)
                continue

            # Transform and save the transformed image and the annotation file.
            transformed_image = apply_neural_style_transfer(cur_content_path,
                                                            cur_style_path,
                                                            args.max_dim,
                                                            weights=args.weights)
            transformed_image.save(cur_out_path)
            cur_anno_out_path = os.path.join(args.out_path, "%s.txt" % cur_content_file_basename)
            shutil.copy(cur_anno_path, cur_anno_out_path)

            if args.grayscale:
                cur_gray_out_path = os.path.join(gray_out_dir, "%s.jpg" % cur_content_file_basename)
                transformed_image.convert('L').save(cur_gray_out_path)
                cur_gray_anno_out_path = os.path.join(gray_out_dir, "%s.txt" % cur_content_file_basename)
                shutil.copy(cur_anno_path, cur_gray_anno_out_path)
        except Exception as e:
            print("Run for content image (in sorted order) %s failed. "
                  "Skipping it and going for the next content image." % c)
            print("Error: %s" % str(e))

    print("Processed Neural Style Transfer for %s images in the content-path and output written to %s"
          % (num_content_images, args.out_path))


if __name__ == "__main__":
    args = get_args()
    process_neural_style_transfer(args)

