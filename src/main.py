import cv2
import numpy as np
from models.face_detection import Face_Detection
from models.facial_landmarks import Facial_Landmark
from models.gaze_estimation import Gaze_Estimation
from models.head_pose_estimation import Head_Pose_Estimation
from mouse_controller import MouseController
from input_feeder import InputFeeder

face_detection_name = "../resources/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
gaze_estimation_name = "../resources/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002"
head_pose_estimation_name = "../resources/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001"
facial_landmarks_name = "../resources/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009"
video = "../bin/How+I+tricked+my+brain+to+like+doing+hard+things.mp4"
image = "../bin/staring.jpg"


def main(media_type, media_path, speed="fast", precision="medium"):

    # Initialize the input feeder
    if media_type == 'video':
        cap = cv2.VideoCapture(media_path)
        width = cap.get(3)  # float
        height = cap.get(4)  # float
        print(width, height)
        #fourcc = cv2.VideoWriter_fourcc(*'MP4')
        #out = cv2.VideWriter('output.mp4', fourcc)
    elif media_type == 'cam':
        cap = cv2.VideoCapture(0)
    else:
        frame = cv2.imread(media_path)


    mouse = MouseController(precision, speed)

    # Initialize and load the gaze estimation model
    gaze_estimation = Gaze_Estimation(gaze_estimation_name)
    gaze_estimation.load_model()
    # Initialize and load the head pose estimation model
    head_pose = Head_Pose_Estimation(head_pose_estimation_name)
    head_pose.load_model()
    # Initialize and load the facial landmarks model
    facial_landmarks = Facial_Landmark(facial_landmarks_name)
    facial_landmarks.load_model()
    # Initialize and model the face detection model
    face_detection = Face_Detection(face_detection_name)
    face_detection.load_model()

    try:
        while True:
            if media_type != 'image':
                ret, frame = cap.read()
            print("entered")
            #frame = batch
            height, width, _ = frame.shape
            print(frame.shape)
            # Process the input image through the face detection first
            # So we can ignore anything that isn't the head
            p_image_f = face_detection.preprocess_input(frame)
            output_f = face_detection.predict(p_image_f)
            f_image = face_detection.preprocess_output(frame, output_f, width, height)
            f_height, f_width, _ = f_image.shape
            print("passed 1")
            # Then we process the output through the head pose estimation
            p_image_h = head_pose.preprocess_input(f_image)
            output_h = head_pose.predict(p_image_h)
            head_position = head_pose.preprocess_output(output_h)
            print("passed 2")
            # And we get the coordinates of the eyes
            p_image_l = facial_landmarks.preprocess_input(f_image)
            output_l = facial_landmarks.predict(p_image_l)
            left_eye_frame, right_eye_frame = facial_landmarks.preprocess_output(output_l, f_image, f_width, f_height)

            # We use the coordinates and the face detection to get an estimate of the gaze
            left_eye = gaze_estimation.preprocess_input(left_eye_frame)
            right_eye = gaze_estimation.preprocess_input(right_eye_frame)
            output_g = gaze_estimation.predict(left_eye, right_eye, head_position)
            print(output_g)


            cv2.imshow("Frame", f_image)
            if cv2.waitKey(10):
                break
    except:
        print("Video has ended or couldn't continue")
    #feed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    media = "video"
    media_file = video
    main(media, media_file)
