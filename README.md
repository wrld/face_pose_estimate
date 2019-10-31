# face_pose_estimate
Estimate the pose of face
## how to estimate the pose of face?
- Detect the face by ssd, using caffe model
- Detect 68 landmarks of face, using caffe model
- Calculate the camera external parameter matrix using pnp
## Environment
- Tengine
- opencv
- C++
## Result
<img src="https://github.com/wrld/face_pose_estimate/master/result.png"/>

<img src="https://github.com/wrld/face_pose_estimate/master/final_result.png"/>
