#include<iostream>
#include<opencv2/opencv.hpp>

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "pose_estimate.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include <sys/time.h>
#include "common.hpp"
int get_current_time1();
struct points{
	int x[68];
	int y[68];
};
class landmark{
public:
	bool first_run=1;
	Mat mean;
	vector<cv::Point2d> my_key_points;
	Mat std;
	graph_t graph;
	Mat origin;
	landmark(std::string protopath, std::string modelpath, std::string meanfile, std::string stdfile);
	~landmark();
	void get_landmark( points pts,cv::Rect offset);
	std::vector<points> get_landmark_batch(std::vector<cv::Mat> imglist, std::vector<cv::Rect> rectlist);
	void get_input_data(cv::Mat& img, float* input_data, int img_h, int img_w);
	


};
void mainloop();
void set_cvMat_input_buffer(std::vector<cv::Mat>& input_channels, float* input_data, const int height, const int width);
