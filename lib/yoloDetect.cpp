#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/all_layers.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "yoloDetect.h"

// constexpr: 상수형 Data 정의, const와 거의 유사함

constexpr float CONFIDENCE_THRESHOLD = 0.7;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};

// auto keyword: 초기화 값에 따라서 알아서 데이터 타입을 정해주는 keyword
// 아래에 NUM_COLORS는 위에 colors[] 배열에 저장된 색깔의 개수를 저장한다.
const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);

void slope_intcpt_from2_pt(double x1, double y1, double x2, double y2, double* m, double* b) {
   *m = (y2 - y1) / (x2 - x1);
   *b = (y1 - ((*m) * x1));
}

bool getState(cv::Point point, double m1, double b1, double m2, double b2) {
   if ((m1 * point.x + b1) < point.y && (m2 * point.x + b2) < point.y && point.y > 280)
      return false;
   else if ((m1 * point.x + b1) == point.y && (m2 * point.x + b2) == point.y && point.y > 280)
      return false;
   else
      return true;
}

void yoloDetect(cv::Mat frame, cv::Mat& result, cv::dnn::Net net, std::vector<std::string> class_names, double m1, double b1, double m2, double b2)
{
    auto output_names = net.getUnconnectedOutLayersNames();
    
    std::vector<cv::Mat> detections; // detections라는 Mat type의 vector정의
    
    cv::Mat blob;

    /*
    chrono 라이브러리에는 두개의 시간 측정 class가 있는데 하나는 Steady_clock, 하나는 system_clock이다.
    system_clock은 전통적인 유닉스 타임(1970 1.1 00:00:00)이후에 흘러간 시간을 뜻하고,
    steady_clock은 pc의 마지막 부팅 이후로 흘러간 시간을 나타낸다.
    보통 일반적인 시간을 나타낼 때 system_clock을 사용하고 timer로 사용할 때는 steady_clock을 사용한다.
     */
    cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false);
    /*
    inference result를 얻어낼 때 model에 이미지를  넣기 전 이미지를 전처리 할때 사용하는 함수가 blobFromImage()이다.
    dnn 모듈에서 1. readNetFrom(network를 읽어오고) 2. blobFromImage (이미지 전처리하고) 3. setInput(전처리된 Image를 입력하고)
    4. forward (forward 연산을 진행한다)

    Network에 이미지가 입력되기 전, blobFromImage에서는 이미지의 사이즈를 통일하고, 4차원 Tensor로 변환하며(N: NumberOfImage, C: NumberOfChannel,
    H: HeightOfImage, W: WidthOfImage)
    blobFromImage(InputImage, OutputImage, scalefactor: image에 곱할 scalar 값, size: outputImage size, Scalar(),
    swapRB: 첫번째 채널과 마지막 채널을 바꾼다 -> RGB2BGR, color영상에서는 꼭 필요함, crop: 이미지를 자른건지 말건지 결정
    ddepth: output blob의 depth -> 보통 CV_32F, CV_8U : image 픽셀의 Type 보통 float 32로 함
    */

    net.setInput(blob); // 처리된 blob을 network의 Input으로 넣는다.

    net.forward(detections, output_names); // network forward연산 -> cv::Mat detection에 detect된 0bject이 저장, output_names에 class이름 저장

    std::vector<int> indices[NUM_CLASSES]; // class 개수 80 크기의 int형 vector 생성
    std::vector<cv::Rect> boxes[NUM_CLASSES]; // class 개수 80 크기의 cv::Rect형 vector 생성 -> bounding box저장
    std::vector<float> scores[NUM_CLASSES]; // class 개수 80 크기의 float형 vector 생성 -> Class 부합도 점수 저장

    for (auto& output : detections) // auto&로 detections들에 접근 -> detections는 Inference type data이기 때문에 auto에 주소 연산자 &를 붙임
    {
        const auto num_boxes = output.rows; // 한 class에 해당되는 Detected object들
        for (int i = 0; i < num_boxes; i++) // box마다 접근 bounding box rect에 저장
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width/2, y - height/2, width, height);

            for (int c = 0; c < NUM_CLASSES; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c); // box정의하고 위에서 정의한 점수보다 높아야 최종적으로 vector에 저장
                if (confidence >= CONFIDENCE_THRESHOLD)
                {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }
    for (int c = 0; c < NUM_CLASSES; c++)
        cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]); // NMS함수: 살아남은 box들 중 NMS함수를 적용하여 한번 더 추림 살아난 box의 Index를 indices에 저장

    for (int c= 0; c < NUM_CLASSES; c++)
    {
        for (size_t i = 0; i < indices[c].size(); ++i)
        {
            const auto color = colors[c % NUM_COLORS]; // class마다 색깔 통일. 4가지 색 중 하나

            auto idx = indices[c][i];
            // 최종 검출된 사각형 객체
            const auto& rect = boxes[c][idx];

            cv::Point p = cv::Point(rect.x+rect.width/2, rect.y+rect.height/2);
            
            if(!getState(p, m1, b1, m2, b2)) {
               //cv::putText(frame, "SAFE", cv::Point(500, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
                cv::putText(frame, "CAUTION", cv::Point(500, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
            /*
            else {
                cv::putText(frame, "CAUTION", cv::Point(500, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
            */

            cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3); // frame에 사각형 그리기

            std::ostringstream label_ss;
            label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
            auto label = label_ss.str();

            int baseline;
            auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
            cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        }
    }
    //float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count(); // 1000 / millisecond 단위의 걸린 시간
    //float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count(); // 1000 / millisecond 단위의 걸린 시간
    /*
    std::ostringstream stats_ss;
    stats_ss << std::fixed << std::setprecision(2);
    stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
    auto stats = stats_ss.str();

    int baseline;
    auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
    */

    result = frame;
    //return frame;
}

