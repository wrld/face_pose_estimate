#include <iomanip>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "landmark.hpp"
#include<iostream>
#include<opencv2/opencv.hpp>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "yu_facedetect.hpp"
#include "pose_estimate.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pose_estimate.h"
#include <sys/time.h>
#include "common.hpp"
#define DEF_PROTO "/home/gjx/文档/ldkproto.prototxt"
#define DEF_MODEL "/home/gjx/文档/ldkmodel.caffemodel"
#define DEF_IMAGE "/home/gjx/图片/选区_025.png"

using namespace std;
using namespace cv;
#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

int get_current_time1()
{
    timeval time;
    gettimeofday(&time,nullptr);
    return (time.tv_sec * 1000 + time.tv_usec/1000);
}

void set_cvMat_input_buffer(std::vector<cv::Mat>& input_channels, float* input_data, const int height, const int width)
{
    for(int i = 0; i < 3; ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }
}

void landmark::get_input_data(cv::Mat& img, float* input_data, int img_h, int img_w)
{
    float* src_ptr=(float*)(img.ptr(0));
    int hw = img_h * img_w;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] =(float)(*src_ptr);
                src_ptr++;
            }
        }
    }
}

landmark::landmark(std::string protopath, std::string modelpath, std::string meanfile, std::string stdfile){
	const char *model_name = "mobilelandmark";
    // this->graph = create_graph(nullptr, "caffe", protopath.c_str(), modelpath.c_str());
    graph_t graph = create_graph(nullptr, "caffe", protopath.c_str(), modelpath.c_str());
    this->graph = graph;
    if(this->graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
    }
	this->mean = cv::imread(meanfile);
    cv::cvtColor(this->mean, this->mean, CV_BGR2RGB);
    cv::resize(this->mean, this->mean,cv::Size(96,96),0,0,cv::INTER_NEAREST);
	(this->mean).convertTo(this->mean, CV_32FC3);
	this->mean = (this->mean)/255.0;
	this->std = cv::imread(stdfile);
    cv::resize(this->std, this->std,cv::Size(96,96),0,0,cv::INTER_NEAREST);
    cv::cvtColor(this->std, this->std, CV_BGR2RGB);
    (this->std).convertTo(this->std, CV_32FC3);
	this->std = (this->std)/255.0;
}

landmark::~landmark(){
    postrun_graph(this->graph);
    destroy_graph(this->graph);
}
void landmark::get_landmark( points pts,cv::Rect offset){
     my_key_points.clear();
     Mat src=origin(offset);
     int ori_h = src.rows;
     int ori_w = src.cols;
     int img_h = 96;
     int img_w = 96;
     cv::Mat resize_img;
     cv::resize(src, resize_img, cv::Size(img_w, img_h), 0, 0,cv::INTER_NEAREST);
     int img_size = img_h * img_w;
     resize_img.convertTo(resize_img, CV_32FC3);
     resize_img = (resize_img/255.0 - this->mean)/(this->std);
   
     float *input_data = (float *)malloc(sizeof(float) * img_size * 3);
     get_input_data(resize_img, input_data, img_h, img_w);


     tensor_t input_tensor = get_graph_input_tensor(this->graph, 0 ,0);
     int dims[] = {1, 3, img_h, img_w};
     set_tensor_shape(input_tensor, dims, 4);
     if (set_tensor_buffer(input_tensor, input_data, img_size * 3 * 4) < 0)
     {
         std::printf("set buffer for input tensor failed\n");
     }
    prerun_graph(this->graph);
    run_graph(this->graph, 1);
    tensor_t output_tensor = get_graph_tensor(this->graph, "fc2");
    float *output = (float *)get_tensor_buffer(output_tensor);
    
     for(int i = 0;i < 135;i =i + 2){
	    pts.x[i / 2] = int(output[i] * ori_w);
 	    pts.y[i / 2] = int(output[i + 1] * ori_h);
        circle(this->origin, cv::Point(pts.x[i / 2]+offset.tl().x,pts.y[i / 2]+offset.tl().y), 2, Scalar(0,0,255), -1);
        my_key_points.push_back(cv::Point2d(pts.x[i / 2]+offset.tl().x ,pts.y[i / 2]+offset.tl().y));
     }

     imshow("origin",origin);
     waitKey(10);
     release_graph_tensor(input_tensor);
     release_graph_tensor(output_tensor);
     free(input_data);
     
}


std::vector<points> landmark::get_landmark_batch(std::vector<cv::Mat> imglist, std::vector<cv::Rect> rectlist){
    int img_h = 96;
    int img_w = 96;
    int img_size = img_h * img_w * 3;
    int batch_size = imglist.size();

    std::vector<points> ptlist;
    if(batch_size == 0){
        return ptlist;
    }
    float *input_data = (float *)malloc(sizeof(float) * img_size * batch_size);
    float *input_data_copy = input_data;
    static bool first_run = true;

    //tensor_t input_tensor = get_graph_input_tensor(this->graph, 0 ,0);
    tensor_t input_tensor = get_graph_tensor(this->graph, "blob1");

    int dims[] = {batch_size, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    if (set_tensor_buffer(input_tensor, input_data, batch_size * img_size * sizeof(float)) < 0)
    {
        std::printf("set buffer for input tensor failed\n");
    }
    for(int i = 0;i < batch_size;i++){
        cv::Mat resized_img;
        cv::resize(imglist[i], resized_img, cv::Size(img_w, img_h), 0, 0,cv::INTER_NEAREST);
        resized_img.convertTo(resized_img, CV_32FC3);
        resized_img = (resized_img/255.0 - this->mean)/(this->std);
        std::vector<cv::Mat> channels;
        set_cvMat_input_buffer(channels, input_data_copy, img_h, img_w);
        cv::split(resized_img, channels);
        input_data_copy = input_data_copy + img_size;
    }
    if(first_run) {
       if(prerun_graph(this->graph) < 0) {
           return ptlist;
       }
       first_run = false;
    }
    if(run_graph(this->graph, 1) != 0)
    {
        std::cout << "Run ONet graph failed, errno: " << get_tengine_errno() << "\n";
    }

    tensor_t output_tensor = get_graph_tensor(this->graph, "fc2");
    float *output = (float *)get_tensor_buffer(output_tensor);

    for(int j = 0;j < batch_size;j++){
        points pts;
        for(int i = 0;i < 135;i =i + 2){
	        pts.x[i / 2] = int(output[i] * imglist[j].cols) + rectlist[j].x;
	        pts.y[i / 2] = int(output[i + 1] * imglist[j].rows) + rectlist[j].y;

        }
        ptlist.push_back(pts);
        output += 136;
    }

    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    return ptlist;
}
int main(int argc, char* argv[])
{
    //初始化参数
    string proto_file;
    string model_file;
    string image_file;
    string stdfile = "/home/gjx/图片/webwxgetmsgimg";
    string meanfile = "/home/gjx/图片/webwxgetmsgimg-1";
    std::string proto_name_ = "/home/gjx/visual-struct/Tengine-master/examples/YuFaceDetectNet/yufacedetectnet-open-v1_new.prototxt";
    std::string mdl_name_ = "/home/gjx/visual-struct/Tengine-master/examples/YuFaceDetectNet/yufacedetectnet-open-v1_new.caffemodel";
    std::string imagefile = "/home/gjx/图片/456.jpeg";
    std::string save_file = "save.jpg";
    vector<float> my_points;
    vector<cv::Point3d> my_model_points;
    Mat pic;
    pic = imread("/home/gjx/图片/选区_061.png");
    ifstream myfile("model.txt");
	string temp;
    //读取人脸标准模型参数
	while(getline(myfile,temp))  
	{
      float f=atof(temp.c_str());
      my_points.push_back(f);
  	} 
    for(int j =0;j<68;j++){
        my_model_points.push_back(cv::Point3d(my_points[j],my_points[j+68],-my_points[136+j]));
    }
    cout<<"my_points"<<my_points.size()<<endl;
    cout<<"my_model_points"<<my_model_points.size()<<endl;
    pose_estimate my_pose(my_model_points);
    //检查参数是否为空
    if(proto_file.empty())
    {
        proto_file = DEF_PROTO;
        std::cout << "proto file not specified,using " << proto_file << " by default\n";
    }
    if(model_file.empty())
    {
        model_file =  DEF_MODEL;
        std::cout << "model file not specified,using " << model_file << " by default\n";
    }
    if(image_file.empty())
    {
        image_file =  DEF_IMAGE;
        std::cout << "image file not specified,using " << image_file << " by default\n";
    }
    Mat img = imread(image_file);
    //初始化tengine
     if(init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }
    if(request_tengine_version("0.9") != 1)
    {
        std::cout << " request tengine version failed\n";
        return 1;
    }
    if(!check_file_exist(proto_file) or (!check_file_exist(model_file) or !check_file_exist(image_file)))
    {
        return 1;
    }
    landmark land_new(proto_file,model_file,meanfile,stdfile);
    VideoCapture capture(0);
    // VideoCapture capture("/home/gjx/视频/VID_20191023_195721.mp4");
    //检查摄像头
    if (!capture.isOpened())
	{
		cout << "some thing wrong" << endl;
		system("pause");
		return -1;
	}

	int  frameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
 
    /*
    ========================================
    for photo detect
    ==========================================
    */
    // Mat frameImg=pic.clone();
    // Mat image;
	// long nCount = 1;
    // face_detect new_face(mdl_name_,proto_name_);
    // points pot;
    // new_face.detect_process(frameImg,save_file);
    // image = new_face.output_image;
    // imshow("detect",image);
    // // waitKey(20000);
    // land_new.origin=image.clone();
    // for(int i = 0;i<new_face.out_rect.size();i++){
    //     land_new.get_landmark(new_face.face_img[i] ,pot,new_face.out_rect[i].tl());
    //     my_pose.my_pose_detect(frameImg,land_new.my_key_points);   
    //     frameImg=my_pose.draw_result(200).clone();
    // }
    // imshow("final",frameImg);
    // waitKey(200000);
    /*===========================================
    for video detect
    =============================================
    */

    Mat frameImg;
    long nCount = 1;
    face_detect new_face(mdl_name_,proto_name_);
    
	while (1){
		capture >> frameImg;
        points pot;
        Mat image;
        //检查摄像头拍摄照片是否为空
        if (!frameImg.empty()){
			imshow("frame", frameImg);
		}
		else{
			continue;
		}
        //人脸检测，并检查人脸检测出的roi是否越界
        if(!new_face.detect_process(frameImg,save_file)){
            continue;
        }
        image = new_face.output_image;
        land_new.origin=image.clone();
        //对于每个roi进行位姿估计
        for(int i = 0;i<new_face.out_rect.size();i++){
        //对人脸关键点检测
        land_new.get_landmark(pot,new_face.out_rect[i]);
        //对人脸位姿估计并画出结果
        my_pose.my_pose_detect(frameImg,land_new.my_key_points);   
        frameImg=my_pose.draw_result(200).clone();
         }
        imshow("final",frameImg);
        waitKey(10);

    	//按Q退出
        if (char(waitKey(40) == 'q')){
			break;
		}
		nCount++;
        
	}
	//释放摄像头
	capture.release();
   	myfile.close();

return 0;
}