#include "pose_estimate.h"

Mat myFaceDetection (Mat src)
{
    Mat src_gray;

    cvtColor(src, src_gray, CV_BGR2GRAY);//转为灰度图
    equalizeHist(src_gray, src_gray);//直方图均衡化，增加对比度方便处理

    CascadeClassifier eye_Classifier;  //载入分类器
    CascadeClassifier face_cascade;    //载入分类器

    //   opencv\sources\data\haarcascades
    if (!eye_Classifier.load("haarcascade_eye.xml")){
        std::cout << "Load haarcascade_eye.xml failed!" << endl;
        return src;
    }

    if (!face_cascade.load("haarcascade_frontalface_alt.xml")){
        std::cout << "Load haarcascade_frontalface_alt failed!" << endl;
        return src;
    }

    //vector 是个类模板 需要提供明确的模板实参 vector<Rect>则是个确定的类 模板的实例化
    std::vector<Rect> eyeRect;
    std::vector<Rect> faceRect;

    //检测关于眼睛部位位置
    eye_Classifier.detectMultiScale(src_gray, eyeRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (size_t eyeIdx = 0; eyeIdx < eyeRect.size(); eyeIdx++){
        rectangle(src, eyeRect[eyeIdx], Scalar(0, 0, 255));   //用矩形画出检测到的位置
    }

    //检测关于脸部位置
    face_cascade.detectMultiScale(src_gray, faceRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (size_t i = 0; i < faceRect.size(); i++){
        rectangle(src, faceRect[i], Scalar(0, 0, 255));      //用矩形画出检测到的位置
    }

    return src;
}
void pose_estimate::my_pose_detect(Mat im,vector<cv::Point2d> image_point){
    this->image_points = image_point;
    // cout<<this->image_points.size();
    dst = im.clone();
    focal_length = im.cols; // Approximate focal length.
    center = cv::Point2d(im.cols/2,im.rows/2);
    camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
    // cout << "Camera Matrix " << endl << camera_matrix << endl ;
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

}
pose_estimate::pose_estimate(vector<cv::Point3d> model_point){
         this->model_points = model_point;
    }
Mat pose_estimate::draw_result(double dist){
    Mat src=dst.clone();
    // for(int i =0;i<68;i++){
    //      result_end_point3D.push_back(cv::Point3d(model_points[i].x*1.2,model_points[i].y*1.2,model_points[i].z*1.2+dist));
    // }
    result_end_point3D.push_back(cv::Point3d(-rear_size,-rear_size,rear_depth));
    result_end_point3D.push_back(cv::Point3d(-rear_size,rear_size,rear_depth));
    result_end_point3D.push_back(cv::Point3d(rear_size,rear_size,rear_depth));
    result_end_point3D.push_back(cv::Point3d(rear_size,-rear_size,rear_depth));
    result_end_point3D.push_back(cv::Point3d(-rear_size,-rear_size,rear_depth));

    result_end_point3D.push_back(cv::Point3d(-front_size,-front_size,front_depth));
    result_end_point3D.push_back(cv::Point3d(-front_size,front_size,front_depth));
    result_end_point3D.push_back(cv::Point3d(front_size,front_size,front_depth));
    result_end_point3D.push_back(cv::Point3d(front_size,-front_size,front_depth));
    result_end_point3D.push_back(cv::Point3d(-front_size,-front_size,front_depth));
    
      // cv::line(src,image_points[0], result_end_point2D[i], cv::Scalar(255,0,0), 2);

    // for(int i = 0;i<model_points.size();i++){
    //     result_end_point3D.push_back(cv::Point3d(model_points[i].x,model_points[i].y,model_points[i].z+dist));
    // } 
   
  
    //  model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
    // model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
    // model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
 
    projectPoints(result_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, result_end_point2D);
//    projectPoints(result_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, result_end_point2D);
   
    for(int i=0; i < image_points.size(); i++)
    {
        circle(src, image_points[i], 2, Scalar(0,0,255), -1);
        // line(src,image_points[i], result_end_point2D[i], cv::Scalar(255,0,0), 2); 

        // cv::line(src,image_points[i], result_end_point2D[i+4], cv::Scalar(255,0,0), 2);
    }
    
     for(int i=0; i < 4; i++)
    {
        // circle(src, image_points[i], 3, Scalar(0,0,255), -1);
        line(src,result_end_point2D[i], result_end_point2D[i+1], cv::Scalar(255,0,0), 2); 
        line(src,result_end_point2D[i+5], result_end_point2D[i+6], cv::Scalar(255,0,0), 2); 
           line(src,result_end_point2D[i], result_end_point2D[i+5], cv::Scalar(255,0,0), 2); 
     
        // cv::line(src,image_points[i], result_end_point2D[i+4], cv::Scalar(255,0,0), 2);
    }
    
   
    // cv::line(src,result_end_point2D[0], result_end_point2D[1], cv::Scalar(255,0,0), 2);
    // cv::line(src,result_end_point2D[1], result_end_point2D[2], cv::Scalar(255,0,0), 2);
    // cv::line(src,result_end_point2D[2], result_end_point2D[3], cv::Scalar(255,0,0), 2);
    // cv::line(src,result_end_point2D[3], result_end_point2D[0], cv::Scalar(255,0,0), 2);
    // cv::line(src,result_end_point2D[4], result_end_point2D[5], cv::Scalar(255,0,0), 2);
    // cv::line(src,result_end_point2D[5], result_end_point2D[6], cv::Scalar(255,0,0), 2);
    // cv::line(src,result_end_point2D[6], result_end_point2D[7], cv::Scalar(255,0,0), 2);
    // cv::line(src,result_end_point2D[7], result_end_point2D[4], cv::Scalar(255,0,0), 2);
    // cv::line(src,result_end_point2D[0], result_end_point2D[4], cv::Scalar(255,255,0), 2);
    // cv::line(src,result_end_point2D[1], result_end_point2D[5], cv::Scalar(255,255,0), 2);
    // cv::line(src,result_end_point2D[2], result_end_point2D[6], cv::Scalar(255,255,0), 2);
    // cv::line(src,result_end_point2D[3], result_end_point2D[7], cv::Scalar(255,255,0), 2);
    
    // cv::line(src,result_end_point2D[], result_end_point2D[0], cv::Scalar(255,0,0), 2);
  
    //draw the calculate point
    // imshow("final",src);
    // waitKey(20000);
    // cout << "Rotation Vector " << endl << rotation_vector << endl;
    // cout << "Translation Vector" << endl << translation_vector << endl;   
    // cout <<  result_end_point2D << endl;
    return src;
    // Display image.
   
}