#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/all_layers.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

void yoloDetect(cv::Mat frame, cv::Mat& result, cv::dnn::Net net, std::vector<std::string> class_names, double m1, double b1, double m2, double b2);
void slope_intcpt_from2_pt(double x1, double y1, double x2, double y2, double* m, double* b);
bool getState(cv::Point point, double m1, double b1, double m2, double b2);
