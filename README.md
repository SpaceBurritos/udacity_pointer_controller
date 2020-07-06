# Computer Pointer Controller

This project consists on the implementation of several ML models so that the pointer on the computer can be controlled
using ones own gaze, this is done with the help of the OpenVino ToolKit

## Project Set Up and Installation
The final structure of the project is the one shown next: 

├─bin  
├─resources  
│  ├─face-detection-adas-binary-0001  
│  │  └─FP32-INT1  
│  ├─gaze-estimation-adas-0002  
│  │  ├─FP16  
│  │  ├─FP16-INT8  
│  │  └─FP32  
│  ├─head-pose-estimation-adas-0001  
│  │  ├─FP16  
│  │  ├─FP16-INT8  
│  │  └─FP32  
│  └─landmarks-regression-retail-0009  
│      ├─FP16  
│      ├─FP16-INT8  
│      └─FP32  
└─src  
│    ├─models  
│    │  ├─face_detection.py  
│    │  ├─facial_landmarks.py  
│    │  ├─gaze_estimation.py  
│    │  └─head_pose_estimation.py  
│    ├─input_feeder.py  
│    ├─main.py  
│    └─mouse_controller.py  
├─README.md  
├─requirements.txt  

The dependencies of this project can be installed on the terminal with `pip3 install -r requirements.txt`

All the ML models used for this project were obtain from the OpenVino model zoo, specifically:

For the Face Detection Model: 
https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html

For the Facial Landmarks Model:
https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html

For the Head Pose Estimation Model:
https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html

For the Gaze Estimation Model:
https://docs.openvinotoolkit.org/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html


When downloading them use the `/resources` folder as the output directory

## Demo
To run the file follow this steps:

* Uncompress the runtime package  
* Install the dependencies present in the `/openvino/install_dependencies` folder of the uncompressed runtime package  
* Source the OpenVINO Environment present in `/bin/setupvars.sh`  
* Type `python main.py --media_file "path/to/file (none if webcam will be used)" --media_type "cam" or "video"`

## Documentation

This project support several command lines:

The `--media_file` is where you specify the path to the media file that you want to use. (This is required)  
The `--media_type` is where you specify the type of media that you are going to use, either: a webcam or a video file (This is required)  
The `--speed` is where you specify the speed of the cursor, the three options are: slow, medium or fast (fast by default)  
The `--precision` is where you specify the precision of the mouse, there are three options: low, medium or high (high by default)  
The `--get_perf_counts` is used to show the time taken by the different parts of the different inference models used 
this can be either true or false (false by default)  
The `--iterations` is used for testing purposes, it iterates over the video file (1 by default)  
The `--batch_size` is the number of frames skipped between the inferences (10 by default)  
The `--show_video` indicates if the video is shown or not, the options are true or false (True by default)  

## Benchmarks

The benchmark results were done using the same hardware for all the tests but changing the models precisions 

**Face Detection:** FP32-INT1  
**Gaze Estimation:** FP16  
**Head Pose Estimation:** FP16  
**Landmarks Regression:** FP16  

**Preprocess Face Detection:** 0.952 ms  
**Inference Face Detection:** 10.84 ms  
**Preprocess Facial Landmark:** 0.0907  
**Inference Facial Landmark:** 1.267 ms  
**Preprocess Head Pose:** 0.090 ms  
**Inference Head Pose:** 0.454 ms  
**Preprocess Gaze Estimation:** 0.0453 ms  
**Inference Gaze Estimation:** 1.67 ms  

- - - -

**Face Detection:** FP32-INT1  
**Gaze Estimation:** FP16 – INT 8  
**Head Pose Estimation:** FP16 – INT 8  
**Landmarks Regression:** FP16 – INT 8  

**Preprocess Face Detection:** 0.794 ms  
**Inference Face Detection:** 11.071 ms  
**Preprocess Facial Landmark:** 0.0678  
**Inference Facial Landmark:** 0.995 ms  
**Preprocess Head Pose:** 0.044736  
**Inference Head Pose:** 0.431 ms  
**Preprocess Gaze Estimation:** 0.0684  
**Inference Gaze Estimation:** 1.22 ms  

- - - -

**Face Detection:** FP32-INT1  
**Gaze Estimation:** FP32  
**Head Pose Estimation:** FP32  
**Landmarks Regression:** FP32  

**Preprocess Face Detection:** 0.813 ms  
**Inference Face Detection:** 11.029 ms  
**Preprocess Facial Landmark:** 0.0739  
**Inference Facial Landmark:** 1.115 ms  
**Preprocess Head Pose:** 0.0739  
**Inference Head Pose:** 0.407 ms  
**Preprocess Gaze Estimation:** 0. ms  
**Inference Gaze Estimation:** 1.66 ms  


## Results

The three benchmarks gave similar results but this could be because in all three of them a CPU was used as the inference
device. If another type of hardware were to be used the difference between the different precisions would be easier to 
spot, having a shorter time the smaller precisions.

## Stand Out Suggestions

Using the option of not showing a video `--show_video false` a difference in the benchmark results could be appreciated:

**Face Detection:** FP32-INT1  
**Gaze Estimation:** FP32  
**Head Pose Estimation:** FP32  
**Landmarks Regression:** FP32  

**Preprocess Face Detection:** 0.175 ms  
**Inference Face Detection:** 8.745 ms  
**Preprocess Facial Landmark:** 0.0581  
**Inference Facial Landmark:** 1.19 ms  
**Preprocess Head Pose:** 0.0191  
**Inference Head Pose:** 0.422 ms  
**Preprocess Gaze Estimation:** 0.115 ms  
**Inference Gaze Estimation:** 1.45 ms 

Specially, the time taken in average on the preprocessing and inference of the Face Detection model was considerably
 smaller

### Edge Cases
Because I didn't have access to a webcam I couldn't optimize for this situation