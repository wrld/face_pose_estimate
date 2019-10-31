/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include "yu_facedetect.hpp"

bool face_detect::post_process_ssd(cv::Mat img, float threshold, float* outdata, int num, std::string save_name)
{
    face_img.clear();
    out_rect.clear();
    std::vector<Box> boxes;
    int line_width = img.cols * 0.005;
    // printf("--------------------------------------------\n");
    // printf("Face id: prob%%\tBOX:( x0 , y0 ),( x1 , y1 )\n");
    // printf("--------------------------------------------\n");
    int detected_face_num = 0;
    for(int i = 0; i < num; i++)
    {
        if(outdata[1] >= threshold)
        {
            detected_face_num += 1;
            Box box;
            box.class_idx = outdata[0];
            box.score = outdata[1];
            box.x0 = outdata[2] * img.cols;
            box.y0 = outdata[3] * img.rows;
            box.x1 = outdata[4] * img.cols;
            box.y1 = outdata[5] * img.rows;
            boxes.push_back(box);
          }
        outdata += 6;
    }
    
    // printf("detect faces : %d \n", detected_face_num);
    for(int i = 0; i < ( int )boxes.size(); i++)
    {
        Box box = boxes[i];
        cv::rectangle(img, cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)), cv::Scalar(255, 255, 0),
                      line_width);

        std::ostringstream score_str;
        score_str.precision(3);
        score_str << box.score;
        std::string label = score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
        cv::rectangle(img,
                      cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
                               cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(img, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
        // face_img.push_back(img(cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0))));
        Rect new_box = cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0));
        out_rect.push_back(cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)));
        if(!(0 <= new_box.x && 0 <= new_box.width && new_box.x + new_box.width <= img.cols && 0 <= new_box.y && 0 <= new_box.height
         && new_box.y + new_box.height <= img.rows)){
        return false;
    }
      }
    output_image = img.clone();
    return true;
}

void face_detect::get_input_data(cv::Mat img, float* input_data, int img_h, int img_w)
{
    int mean[3] = { 104,117,123 };
    unsigned char* src_ptr=(unsigned char*)(img.ptr(0));
    int hw = img_h * img_w;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] =(float)(*src_ptr - mean[c]);
                src_ptr++;
            }
        }
    }
}
bool face_detect::detect_process(Mat image_file,string save_file){
  cv::Mat img = image_file.clone();
    if(img.empty())
    {
        std::cerr << "failed to read image file " << image_file << "\n";
    }
#if 1
    // resize to 320 x 240



    cv::Mat resize_img;
    // int img_w = 320;
    // int img_h = 240;
    cv::resize(img, resize_img, cv::Size(img_w, img_h), 0, 0,cv::INTER_NEAREST);
    input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);
    get_input_data(resize_img, input_data, img_h, img_w);
#else
    // use origin image size
    int img_h = img.rows;
    int img_w = img.cols;
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);
    get_input_data(img, input_data, img_h, img_w);
#endif
 /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    /* setup input buffer */
    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
    
    }

    prerun_graph(graph);

    // time run_graph
    int repeat_count = 1;
    char* repeat = std::getenv("REPEAT_COUNT");
    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    struct timeval t0, t1;
    float avg_time = 0.f;
    gettimeofday(&t0, NULL);
    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);
    gettimeofday(&t1, NULL);
    float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    avg_time += mytime;
    // std::cout << "--------------------------------------\n";
    // std::cout << "repeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";

    // post process
    this->out_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out");
    int out_dim[4];
    get_tensor_shape(out_tensor, out_dim, 4);
    float* outdata = ( float* )get_tensor_buffer(out_tensor);
    int num = out_dim[1];
    return post_process_ssd(img, show_threshold, outdata, num, save_file.c_str());
}

face_detect::face_detect(string mdl_name_,string proto_name_ ){
    init_tengine();
    this->graph = create_graph(nullptr, "caffe", proto_name_.c_str(), mdl_name_.c_str());
    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
      
    }   
}

face_detect::~face_detect(){
    release_graph_tensor(out_tensor);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
    release_tengine();
}

