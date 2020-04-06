# Synthetic Underwater Object Detection
A proof-of-concept research project at Applied Physics Lab at University of Washington.
Can we detect an object underwater using a model that only knows how the object looks in a 3D model (CAD/Solidworkds)?

Data and evaluation directory can be requested by contacting us.

## Motivation
In certain scenarios, object detection systems are expected to work in real-world scenes with no or very-limited prior in-situ (real-world) exposure. We have a scenario where we expect underwater bots with mono camera (greyscale) to detect an object located underwater in scenes to which we have no or very limited prior exposure. Studying the impact of in-situ data on the performance of such a system would help us plan out the data collection requirements to deploy the system in real-world environments.

## Literature Review
Prior literature have experimented with various methods to work with synthetic data or produce synthetic data which ensures

- Neural Style Transfer [[1]](https://arxiv.org/abs/1508.06576): Using a pre-trained network, we can extract the feature maps of the content image (a sample image containing the object) and style image separately, and get an image where the style gets fused into the content image.

- Learning Deep Object Detection from 3D Models [[2]](https://arxiv.org/abs/1412.7122): A study showing the use of synthetic non-photorealistic images rendered from 3D models for object detection on Imagenet classes. Shows that using synthetic data can help in improving the average precision of the object detection model. It makes it possible to have detection for classes have no real training data. As the number of real training data increases, the average precision increases and starts to saturate for upto 20 real images.

- Synthetic to Real Adaptation with Generative Correlation Alignment Networks [[3]](https://arxiv.org/abs/1701.05524): Shows that using neural style transfer gives a lift in average precision on imagenet classes compared to using just synthetic non-photorealistic images from [[2]](https://arxiv.org/abs/1412.7122). Also proposes DGCAN which performs slightly better than neural style transfer. Implementation of DGCAN [[4]](https://arxiv.org/abs/1511.06434) present in Caffe, and currently an implementation is also present in TensorFlow.

## Problem Statement
Our goal is to generate synthetic training data for object detection given a 3D model of the object of interest. Also, we aim to study the effect of adding in-situ data on the performance of such an object detection method.

We consider a test set from real-world scenarios consisting of images with labeled bounding boxes over the object of interest. We consider Average Precision at 0.5 IOU (Intersection Over Union) as primary evaluation metric. We compare the performance of various settings of object detection models:
- Built with only Synthetic data. <br>
- Built with only Synthetic data, plus 50 in-situ background images. <br>
- Built with only Synthetic data, plus 50 in-situ background images, plus 10 in-situ background images containing real-world object of interest.

## Methodology
We use a simple methodology takes the 3D model of the object (.egg file) to produce synthetic training data images, which are further used to train a YOLOv3 [[5]](https://arxiv.org/abs/1804.02767) object detection model using Darknet [[6]](https://pjreddie.com/darknet/), which in turn is evaluated on a test set. We mainly focus on the process of producing the synthetic training data images in this study. The details on the training and evaluation procedure for YOLOv3 object detection model are described on the Darknet website [[6]](https://pjreddie.com/darknet/).

The below Flowchart provides an overview of different modules of our methodology.

![Flowchart](https://github.com/bhuvi3/underwater_synthetic_image_recognition/blob/master/underwater_synthetic_object_detection-flowchart.png) 

##### Baseline Model
As a baseline method to produce synthetic training data, we utilize a method similar [[2]](https://arxiv.org/abs/1412.7122), where we generate non-photorealistic synthetic data by rendering from the 3D model onto an underwater background. This process has been depicted in *Image Renderer* module in the above flowchart. It takes the 3D model along with a set of background images as input and does a simple overlay of the object onto each background image at different positions, distances (zoom levels) and lightings.

For the baseline model, we use ~250 underwater images from [Unsplash](https://unsplash.com/about) which are freely available online on [Unsplash license](https://unsplash.com/license), as background images. We produce a training data with ~1000 images containing overlaid objects with random position, zoom and lighting. Note that each image is randomly chosen to contain 0, 1 or 2 object instances. Qualitatively, this results in a training set containing non-photorealistic images.

As a final step, the rendered images are converted to greyscale (since our in-situ images are captured by a mono camera which produces greyscale images). The images are also resized to have a maximum dimension of 256 retaining the aspect ratio. Hence, we produce labels for our synthetic training data which is used to train a YOLOv3 [[5]](https://arxiv.org/abs/1804.02767) object detection model using Darknet.

##### NST Model
In this study, we test whether NST helps in improving the quality of our rendered non-photorealistic images, to make them more photo-realistic so that they are similar to in-situ images. Therefore, currently we use a standard implementation of *NST model* with VGG-19 base architecture for the feature extraction network which is pretrained on ImageNet. This module takes a pair of content-image and style-image to produce a NST version of the content-image. Qualitatively, this process produces a better blend of the object onto the background, but results in small artifacts spread across the image.

This module takes the rendered non-photorealistic training images (produced for the baseline model as explained above) as input, and considers each image as a content-image and its corresponding original background image (a greyscale version of the original background image needs to be considered if it is a color image, since rendered images are converted to greyscale) as the style-image, to produce the NST version of the synthetic training data for the YOLOv3 object detection model.

##### Experimental Settings
We test the effectiveness of the NST in resulting improvement in the object detection model in different experimental settings. In each setting, we compare the Baseline Model against NST model on the test set using Average Precision at 0.5 IOU metric. These different settings help us determine the effectiveness of including in-situ data (background only, and background with real-world object).

- **Unsplash-Insitu-0 :** Training data contains only synthetic background images from Unsplash, and contains no in-situ data at all.

- **Unsplash-Insitu-50 :** Training data contains synthetic background images from Unsplash along with 50 in-situ background images.  200 scenes are produced from these 50 background images by overlaying objects at random positions, zoom-levels and lighting. Note that this does not contain any images with in-situ real-world object, but it contains only in-situ backgrounds on which 3D models are overlaid as described in the Baseline Model section.

- **Unsplash-Insitu-50-RealObject10 :** Training data contains synthetic background images from Unsplash along with 50 in-situ background images, and it also includes 10 in-situ images containing real-world object. These 10 real-object images were repeated 20 times resulting in 200 such scenes, to increase their prevalence in the training data.

**Note:** We did not pursue the generation of artificial background images using DCGAN [[4]](https://arxiv.org/abs/1511.06434) using synthetic images, because we consider a setting where we do not have real-world background images. We experimented using DCGAN to generate synthetic images using synthetic underwater images, but we observed qualitatively insufficient results, and it was computationally very expensive as well.

## Results and Discussion
The experimental results are based on the test set containing 313 in-situ images some of which contain real-world object of interest. This test set does not include complicated cases like flash lighting, turbid water or close-up object. The table below contains the Average Precision at 0.5 IOU (mentioned as a percentage) on this test set by both baseline and the NST models in the 3 different settings explained in the Methodology section.

| Model-Type     | Unsplash-Insitu-0 | Unsplash-Insitu-50 | Unsplash-Insitu-50-RealObject10 |
|----------------|-------------------|--------------------|---------------------------------|
| Baseline Model | 72.51%            | 72.53%             | 77.66%                          |
| NST Model      | 71.80%            | 75.69%             | 79.41%                          |

We can observe that we are able to achieve decent performance of ~72% Average Precision on the test set with the Baseline Model without consuming any in-situ data during the training time.

We can observe that *NST Model* does not seem to be useful for the setting which contains no in-situ background images. However, as we add scenes from 50 in-situ background images, *Baseline Model* performance remains almost unchanged, but NST model provides ~3% improvement over the Baseline Model. Adding in-situ images containing real-world object provides considerable improvement in both Baseline Model and NST Model, and NST Model provides ~2% better performance than Baseline Model.

Few other experiments were conducted by adding varied numbers (50, 100, 150, 200, 250) of in-situ background images (without real-world objects), and we observed that the NST Model performs better than the Baseline Model as more in-situ data is added.

A limitation of our study is that the in-situ images added in *Unsplash-Insitu-50* and *Unsplash-Insitu-50-RealObject10* settings were captured during the same time as we captured our test-set, hence the in-situ images in these settings are very similar to the images in the test set.

**Note:** We tested by training with color-images as well. However, we did not have the color-images in the test data, hence they have been discussed in this brief report. These results are noted in the linked performance analysis sheet.

## Conclusion and Future Work
Our study shows that the simple rendering of objects on several underwater background images from Unsplash, without using any in-situ data is able to provide a decent performance of ~72% (Baseline Model), and NST Model does not help in providing performance improvement in this setting. Also, we observed a considerable performance improvement in both Baseline Model and NST Model when we added in-situ background images and in-situ images containing real-world object. However, the NST Model was found to perform considerably better than the Baseline Model in settings which included in-situ background images and in-situ images with real-world object. Note that these inferences are based on Average Precision as a metric.

This implies that we can deploy such object detections systems in our underwater environments without any in-situ data during training time, but we can expect to improve its performance using NST Model by ~7% when we are able to procure some (~50) in-situ background images and very few (~10) in-situ images with real-world object.

Future work can consider experimenting with variation of textures in rendering the synthetic images, which may have an effect on NST. Currently, we use NST with VGG-19 base architecture for feature extraction which is pre-trained on Image-Net. Future work can consider experimenting with different base architecture like ResNets, and use a different dataset to pre-train such feature extraction networks. This would customize the NST to our problem setting with underwater backgrounds. As discussed before, we could use a generative method to produce the backgrounds, and hence future work can build such a generative model like DCGAN on several in-situ backgrounds, to generate as many underwater images with variations in depth, lighting, turbidity and other aspects of an underwater scene. Our target scene is an underwater image taken in a mono-camera (greyscale), but currently we utilize color-images to produce the synthetic images. Therefore, using more data from a mono-camera to procure the background images for the image rendering process, may also be effective.

## References
[[1] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).](https://arxiv.org/abs/1508.06576)

[[2] Peng, Xingchao, et al. "Learning deep object detectors from 3d models." Proceedings of the IEEE International Conference on Computer Vision. 2015.](https://arxiv.org/abs/1412.7122)

[[3] Peng, Xingchao, and Kate Saenko. "Synthetic to real adaptation with generative correlation alignment networks." 2018 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2018.](https://arxiv.org/abs/1701.05524)

[[4] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).](https://arxiv.org/abs/1511.06434)

[[5] Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).](https://arxiv.org/abs/1804.02767)

[[6] Redmon, Joseph. "Darknet: Open Source Neural Networks in C." web (2013-2016).](https://pjreddie.com/darknet/)
