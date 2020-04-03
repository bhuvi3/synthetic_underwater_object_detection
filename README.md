# Synthetic Underwater Object Detection
A proof-of-concept research project at Applied Physics Lab at University of Washington.
Can we detect an object underwater using a model that only knows how the object looks in a 3D model (CAD/Solidworkds)?

Data and evaluation directory can be requested by contacting us.

## Motivation
In certain scenarios, object detection systems are expected to work in real-world scenes with no or very-limited prior in-situ (real-world) exposure. We have a scenario where we expect underwater bots with greyscale camera to detect an object located underwaater in scenes to which we have no or very limited prior exposure. Studying the impact of in-situ data on the performance of such a system would help us plan out the data collection requirements to deploy the system in real-world environments.

## Literature Review
Prior literature have experimented with various methods to work with synthetic data or produce synthetic data which ensures

- Neural Style Transfer [[1]](https://arxiv.org/abs/1508.06576): Using a pre-trained network, we can extract the feature maps of the content image (a sample image containing the object) and style image separately, and get an image where the style gets fused into the content image.

- Learning Deep Object Detection from 3D Models [[2]](https://arxiv.org/abs/1412.7122): A study showing the use of synthetic non-photorealistic images rendered from 3D models for object detection on Imagenet classes. Shows that using synthetic data can help in improving the average precision of the object detection model. It makes it possible to have detection for classes have no real training data. As the number of real training data increases, the average precision increases and starts to saturate for upto 20 real images.

- Synthetic to Real Adaptation with Generative Correlation Alignment Networks [[3]](https://arxiv.org/abs/1701.05524): Shows that using neural style transfer gives a lift in average precision on imagenet classes compared to using just synthetic non-photorealistic images from [[2]](https://arxiv.org/abs/1412.7122). Also proposes DGCAN which performs slightly better than neural style transfer. Implementation of DGCAN [[4]](https://arxiv.org/abs/1511.06434) present in Caffe, and currently an implementation is also present in TensorFlow.

## Problem Statement
Our goal is to generate synthetic training data for object detection given a 3D model of the object of interest. Also, we aim to study the effect of adding in-situ data on the performance of such an object detection method.

We consider a test set from real-world scenarios consisting of images with labeled bounding boxes over the object of interest. We consider Average Precision at 0.5 IOU (Intersection Over Union) as primary evaluation metric. We compare the performance of various settings of object detection models:
- Built with only Synethitc data. <br>
- Built with only Synethitc data, plus 50 in-situ background images. <br>
- Built with only Synethitc data, plus 50 in-situ background images, plus 10 in-situ background images containing real-world object of interest.

## Methodology
TODO

**Note:** We did not pursue with generation of artificial background images using DCGAN [[4]](https://arxiv.org/abs/1511.06434) using synthetic images, because we consider a setting where we do not have real-world background images. We experimented using DCGAN to generate synthetic images using synthetic underwated images, but we observed qualitatively insufficient results, and it was computationally very expensive as well. 

## Results and Discussion
TODO

**Note:** We tested by training with color-images as well. However, we did not have the color-images in the test data, hence they have been discussed in this brief report. These results are noted in the linked performance analysis sheet.

## Conclusion and Future Work
TODO


## References
[[1] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).](https://arxiv.org/abs/1508.06576)

[[2] Peng, Xingchao, et al. "Learning deep object detectors from 3d models." Proceedings of the IEEE International Conference on Computer Vision. 2015.](https://arxiv.org/abs/1412.7122)

[[3] Peng, Xingchao, and Kate Saenko. "Synthetic to real adaptation with generative correlation alignment networks." 2018 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2018.](https://arxiv.org/abs/1701.05524)

[[4] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).](https://arxiv.org/abs/1511.06434)
