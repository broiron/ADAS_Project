# Simple ADAS Project
In this project, We developed ADAS(Advanced Driver Assistance System) by using C++ language.

## Inspired by
- https://github.com/ultralytics/yolov3

## Project environment
- Ubuntu 20.04.3 LTS
- OpenCV 4.5.5
- GeForce GTX 1060 6GB
- CMake 3.16.3
- Used USB video camera (resolution: 640 * 480)

## Project mechanism
#### Lane detection
1. Convert video's color space from BGR to HSV.
2. In HSV Color space, by using `cv::inRange()`, we extract the images that have the same color as the lane.
3. Pass the low-pass filter to remove noise.
4. Do dilation operation to make lane more clear in the image and then extract the lane by using `Canny edge` and `HoughLines Transform`.
5.Lane departure warning alert when the driving car escape the detected lane.

#### Object detection
1. Used yoloV3-tiny pre-trained weight (COCO) for car & other object detection.
2. If the detected object's location is located within the pre-specified ROI(In front of the car), collision dager alert perform.

#### Features
- This program shows us whether the driver keeps his lane and a safe distance.
- Upper left side: Print Safe if the driver stay in his lane. Print Warning if it's not.
- Upper right side: Print Safe if the driver keep a safe distance. Print Warning if it's not.
