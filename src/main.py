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
video = "../bin/demo.avi"
image = "../bin/staring.jpg"


def main(media_type, media_path, speed="fast", precision="high"):

    # Initialize the input feeder
    '''
    if media_type == 'video':
        cap = cv2.VideoCapture(media_path)
        width = cap.get(3)
        height = cap.get(4)


    elif media_type == 'cam':
        cap = cv2.VideoCapture(0)
        width = cap.get(3)
        height = cap.get(4)
    else:
        frame = cv2.imread(media_path)
        height, width, _ = frame.shape
    '''
    mouse = MouseController(precision, speed)
    feed = InputFeeder(media_type, media_path)
    feed.load_data()

    if media_type != "image":
        width = feed.cap.get(3)
        height = feed.cap.get(4)
    else:
        height, width, _ = feed.cap.shape

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
        for frame in feed.next_batch(media_type):
            #print(frame)
            # if media_type != 'image':
            #    ret, frame = cap.read()

            # Process the input image through the face detection first
            # So we can ignore anything that isn't the head
            p_image_f = face_detection.preprocess_input(frame)
            output_f = face_detection.predict(p_image_f)
            f_image = face_detection.preprocess_output(frame, output_f, width, height)
            f_height, f_width, _ = f_image.shape

            # Then we process the output through the head pose estimation
            p_image_h = head_pose.preprocess_input(f_image)
            output_h = head_pose.predict(p_image_h)
            head_position = head_pose.preprocess_output(output_h)

            # And we get the coordinates of the eyes
            p_image_l = facial_landmarks.preprocess_input(f_image)
            output_l = facial_landmarks.predict(p_image_l)
            left_eye_frame, right_eye_frame = facial_landmarks.preprocess_output(output_l, f_image, f_width, f_height)

            # We use the coordinates and the face detection to get an estimate of the gaze
            left_eye = gaze_estimation.preprocess_input(left_eye_frame)
            right_eye = gaze_estimation.preprocess_input(right_eye_frame)
            output_g = gaze_estimation.predict(left_eye, right_eye, head_position)

            x, y, gaze_vector = gaze_estimation.preprocess_output(output_g)
            print(x,y)
           # mouse.move(x,y)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
    except:
        print("Video has ended or couldn't continue")
    #feed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    media = "image"
    media_file = image
    main(media, media_file)
