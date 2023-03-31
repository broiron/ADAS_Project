#include "yoloDetect.h"
#include <iostream>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "laneDetect.h"

int main() 
{   
    
   double x[] = { 400, 220, 600, 390 };
   double y[] = { 300, 420, 420, 300 };

    double m1, b1, m2, b2;
    
    // ROI에 대한 기울기 받아오기
    slope_intcpt_from2_pt(x[0], y[0], x[1], y[1], &m1, &b1);
    slope_intcpt_from2_pt(x[2], y[2], x[3], y[3], &m2, &b2);


    cv::VideoCapture cap("./img/test.mp4");
    //cv::VideoCapture cap("/dev/video0", cv::CAP_V4L);
	if (!cap.isOpened()) {
		printf("Can't open the camera\n");
		return -1;
	}
    
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("./yolo/yolov3-tiny.cfg", "./yolo/yolov3-tiny.weights"); // cfg, weights 읽어오고 yolo network불러오기
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA); // 네트워크 백엔드 지정
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); // 네트워크가 선호하는 타겟 디바이스 지정 -> python에서의 device정의와 유사

    std::vector<std::string> class_names;
    std::ifstream class_file("./src/classes.txt"); // ifstream: Input file stream -> 인자로 들어가는 파일을 읽고 이를 class_file에 저장한다.
        if (!class_file)
        {
            std::cerr << "failed to open classes.txt\n";
            exit(1);
        }

        std::string line;
        while (std::getline(class_file, line)) // 각 라인을 읽고 이를 위에서 정의한 classes_names vector에 하나씩 저장한다.
            class_names.push_back(line);
    cv::Mat img;
    cv::Mat img_;
    cv::Mat img_result;
    std::vector<Point> lane;
    printf("Before Processing\n");
	while (1) {
        auto total_start = std::chrono::steady_clock::now();
		//cap >> img;
        cap >> img_;

        cv::resize(img_, img, cv::Size(640, 480));

		if (img.empty()) {
			printf("empty image");
			return -1;
		}
		
        auto dnn_start = std::chrono::steady_clock::now();

        lane_detection(img, lane);
        yoloDetect(img, img_result, net, class_names, m1, b1, m2, b2);
        
        img_result = drawLine(img_result, lane);
        img_result = Lane_warning(img_result, lane);

        auto dnn_end = std::chrono::steady_clock::now();
        auto total_end = std::chrono::steady_clock::now();
		float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(img_result, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(img_result, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
        cv::imshow("result", img_result);
        
		if (cv::waitKey(1) == 27)
			break;
    }
    
    return 0;
}

