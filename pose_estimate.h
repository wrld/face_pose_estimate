#ifndef POSE_ESTIMATE
#define POSE_ESTIMATE
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;
 
class pose_estimate{
public:
    std::vector<cv::Point2d> image_points;
    std::vector<cv::Point3d> model_points;
    Mat dst;
    double focal_length;
    Point2d center; 
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::Mat rotation_vector; 
    double rear_size = 75;
    double rear_depth = 0;
    double front_size = 100;
    double front_depth = 200;
    cv::Mat translation_vector;
    vector<Point3d> result_end_point3D;
    vector<Point2d> result_end_point2D;
    vector<Point3d> box_model3D;
    void my_pose_detect(Mat im,vector<cv::Point2d> image_point);
    pose_estimate(vector<cv::Point3d> model_point);
    // Mat myFaceDetection (Mat src);
    Mat draw_result(double dist);
    Mat myFaceDetection (Mat src);
};

#endif