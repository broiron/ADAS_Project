#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "laneDetect.h"

double img_center;
double left_m, right_m;
Point left_b, right_b;
bool left_detect = false, right_detect = false;

vector<vector<Vec4i>> separateLine(Mat img_edges, vector<Vec4i> lines);
vector<Point> regression(vector<vector<Vec4i>> separatedLines, Mat img_input);
Mat drawLine(Mat img_input, vector<Point> lane);
Mat Lane_warning(Mat img_input, vector<Point> lane);

void lane_detection(Mat img, vector<Point>& lane) {

	Mat HSV;

	cvtColor(img, HSV, COLOR_BGR2HSV); //Color space change
	Scalar lower_white = Scalar(0, 0, 200);
	Scalar upper_white = Scalar(180, 255, 255);
	inRange(HSV, lower_white, upper_white, HSV); 
	Mat bilateral_Filter, dilate_filter, canny;
	bilateralFilter(HSV, bilateral_Filter, 5, 100, 100); 
	dilate(bilateral_Filter, dilate_filter, Mat()); 
	Canny(dilate_filter, canny, 150, 255);

	//ROI
    
    Point point[4];     
    point[0] = Point(270, 330);
    point[1] = Point(95, 430);
    point[2] = Point(560, 430);
    point[3] = Point(370, 330);
    

	Mat img_mask = Mat::zeros(canny.rows, canny.cols, CV_8UC1);
	Mat ROI = Mat::zeros(canny.rows, canny.cols, CV_8UC1);

	Scalar ignore_mask_color = Scalar(255, 255, 255);
	const Point* ppt[1] = { point };
	int npt[] = { 4 };

	fillPoly(img_mask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);

	bitwise_and(canny, img_mask, ROI);

	Mat lineResult;
	cvtColor(ROI, lineResult, COLOR_GRAY2BGR); 
    //imshow("roi", ROI);

	vector<Vec4i> lines; 
	HoughLinesP(ROI, lines, 1, CV_PI / 180, 30, 10, 20);

	vector<vector<Vec4i> > separated_lines;
	
    if (lines.size() > 0) {
		separated_lines = separateLine(ROI, lines);
		lane = regression(separated_lines, img);
	}
    
    else {
        Point p = Point(-100, -100);
        lane = { p, p, p, p };
    }      
}

Mat Lane_warning(Mat img_input, vector<Point> lane) {
   Vec2i R1 = lane[0];
   Vec2i R2 = lane[1];
   Vec2i L1 = lane[2];
   Vec2i L2 = lane[3];

   printf("R1: (%d, %d)", R1[0], R1[1]);
   printf(" R2: (%d, %d)", R2[0], R2[1]);
   printf(" L1: (%d, %d)", L1[0], L1[1]);
   printf(" L2: (%d, %d)\n", L2[0], L2[1]);

   double lane_center = (R1[0] + L1[0]) / 2;
   printf("lane_center %f\n", lane_center);
   printf("lane_center %f\n", img_center);

   putText(img_input, "|", Point(img_center, 450), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);
   putText(img_input, "|", Point(lane_center, 450), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1, LINE_AA);

   if (lane_center < (img_center - 100)) {
      printf("Warning\n");
      putText(img_input, "CAUTION", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
   }
   else if (lane_center > (img_center + 100)) {
      printf("Warning\n");
      putText(img_input, "CAUTION", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
   }
   else {
      printf("Safe\n");
      //putText(img_input, "SAFE", Point(20, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, LINE_AA);
   }

   return img_input;
}

vector<vector<Vec4i>> separateLine(Mat img_edges, vector<Vec4i> lines) {

	vector<vector<Vec4i>> output(2);
	Point ini, fini;
	vector<double> slopes;
	vector<Vec4i> selected_lines, left_lines, right_lines;

	double slope_thresh = 0.3;

	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		ini = Point(line[0], line[1]);
		fini = Point(line[2], line[3]);

		double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y))
			/ (static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001); 

		if (abs(slope) > slope_thresh) {
			slopes.push_back(slope); 
			selected_lines.push_back(line); 
		}
	}


	img_center = static_cast<double>((img_edges.cols / 2));
	for (int i = 0; i < selected_lines.size(); i++) {
		ini = Point(selected_lines[i][0], selected_lines[i][1]);
		fini = Point(selected_lines[i][2], selected_lines[i][3]);

		if (slopes[i] > 0 && fini.x > img_center && ini.x > img_center) {
			right_lines.push_back(selected_lines[i]);
			right_detect = true;
		}
		else if (slopes[i] < 0 && fini.x < img_center && ini.x < img_center) {
			left_lines.push_back(selected_lines[i]);
			left_detect = true;
		}
	}

	output[0] = right_lines;
	output[1] = left_lines;
	return output;
}

vector<Point> regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {

	vector<Point> output(4);
	Point ini, fini;
	Point ini2, fini2;
	Vec4d left_line, right_line; 
	vector<Point> left_pts, right_pts;

    int ini_y = img_input.rows;
    int fin_y = 350;

    double right_ini_x, right_fin_x, left_ini_x, left_fin_x;

	if (right_detect) {
		for (auto i : separatedLines[0]) { //output[0] = right_lines
			ini = Point(i[0], i[1]);
			fini = Point(i[2], i[3]);

			right_pts.push_back(ini);
			right_pts.push_back(fini);
		}
		if (right_pts.size() > 0) {
			fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01); 

			right_m = right_line[1] / right_line[0];  //����
			right_b = Point(right_line[2], right_line[3]); //������ �� ��
		}

        right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
        right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;
	}

    else {
        right_ini_x = -100;
        right_fin_x = -100;
    }

	if (left_detect) {
		for (auto j : separatedLines[1]) { //output[1] = left_lines
			ini2 = Point(j[0], j[1]);
			fini2 = Point(j[2], j[3]);

			left_pts.push_back(ini2);
			left_pts.push_back(fini2);
		}
		if (left_pts.size() > 0) {
			fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];  //����
			left_b = Point(left_line[2], left_line[3]);
		}

        left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
        left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;
	}

    else {
        left_ini_x = -100;
        left_fin_x = -100;
    }

	output[0] = Point(right_ini_x, ini_y);
	output[1] = Point(right_fin_x, fin_y);
	output[2] = Point(left_ini_x, ini_y);
	output[3] = Point(left_fin_x, fin_y);

	return output;
}

Mat drawLine(Mat img_input, vector<Point> lane) {
    
    vector<Point> points;
    Mat output;

    img_input.copyTo(output);
    points.push_back(lane[2]);
    points.push_back(lane[0]);
    points.push_back(lane[1]);
    points.push_back(lane[3]);

    if (lane[0].x > 0 && lane[2].x > 0) {
        fillConvexPoly(output, points, Scalar(0, 255, 0), LINE_AA, 0);
        addWeighted(output, 0.2, img_input, 0.8, 0, img_input);
    }

	return img_input;
}



