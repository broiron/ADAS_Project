# Simple ADAS Project
In this project, We developed ADAS(Advanced Driver Assistance System) by using C++ language.

## Inspired by


## Project environment
- Ubuntu 20.04.3 LTS
- OpenCV 4.5.5
- GeForce GTX 1060 6GB
- CMake 3.16.3
- Used USB video camera (resolution: 640 * 480)

## Project background
#### Lane detection
1. Convert video's color space from BGR to HSV.
2. In HSV Color space, by using cv::inRange(), we extract the images that have the same color as the lane.
3. Pass the low-pass filter to remove noise.
4. Do dilation operation to make lane more clear in the image and then extract the lane by using Canny edge.



#### Object detection


- Used yoloV3-tiny for car detection
- Used Hough transform edge detection for lane detection

- Developed in Desktop (GPU: GTX1080) & Jetson-nano
