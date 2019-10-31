#ifndef FACEDETECTYU
#define FACEDETECTYU

#include <iostream>
#include <iomanip>
#include <sys/time.h>

#include "tengine_c_api.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};
class face_detect{
public:
float show_threshold = 0.9;
 int img_w = 320;
    int img_h = 240;
    graph_t graph;
	float* outdata;
    tensor_t out_tensor;
    tensor_t input_tensor;
    int num;
    Mat output_image;
    vector<Mat> face_img;
    vector<Rect> out_rect; 
    float* input_data;
    ~face_detect();
    const std::string save_name;
    bool detect_process(Mat image_file,string save_file);
	face_detect(string mdl_name_,string proto_name_);
	bool post_process_ssd(cv::Mat img, float threshold, float* outdata, int num, string save_name);
    void get_input_data(cv::Mat img, float* input_data, int img_h, int img_w);

};

#endif