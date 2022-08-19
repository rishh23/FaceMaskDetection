# Face-Mask-Detector
Real time face-mask detection using Deep Learning and OpenCV

# About Project

This project uses a Deep Neural Network, more specifically a Convolutional Neural Network, to differentiate between images of people with and without masks. The CNN manages to get an accuracy of 99.12% on the training set and 95.88% on the test set. Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using OpenCV. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task. The model works efficiently with no apparent lag time between wearing/removing mask and display of prediction.

The model is capable of predicting multiple faces with or without masks at the same time

# Dataset
The data used can be downloaded through this https://data-flair.training/blogs/download-face-mask-data/ . There are 1314 training images and 194 test images divided into two catgories, with and without mask.


