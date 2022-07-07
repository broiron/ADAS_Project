#pragma once

#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

void lane_detection(Mat img, vector<Point>& lane);
Mat Lane_warning(Mat img_input, vector<Point> lane);
Mat drawLine(Mat img_input, vector<Point> lane);

