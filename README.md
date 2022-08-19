# Face-Mask-Detector
Real time face-mask detection using Deep Learning and OpenCV

# About Project

This project uses a Deep Neural Network, more specifically a Convolutional Neural Network, to differentiate between images of people with and without masks. The CNN manages to get an accuracy of 99.12% on the training set and 95.88% on the test set. Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using OpenCV. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task. The model works efficiently with no apparent lag time between wearing/removing mask and display of prediction.

The model is capable of predicting multiple faces with or without masks at the same time

# Working
With Mask

![Screenshot (39)](https://user-images.githubusercontent.com/66566396/124602193-3b7d8d80-de86-11eb-8156-e94c6f6d8d67.png)


No Mask

![Screenshot (40)](https://user-images.githubusercontent.com/66566396/124602348-5ea83d00-de86-11eb-90c0-6baa5c2b8c67.png)


# Dataset
The data used can be downloaded through this https://data-flair.training/blogs/download-face-mask-data/ . There are 1314 training images and 194 test images divided into two catgories, with and without mask.

How to Use
To use this project on your system, follow these steps:

1.Clone this repository onto your system by typing the following command on your Command Prompt:
git clone https://github.com/Sudarshan201125/Face-Mask-Detector.git

2.cd FaceMaskDetector

3.Run facemask.py by typing the following command on your Command Prompt:
py -m facemask.py

The Project is now ready to use !!
