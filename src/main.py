import cv2
import numpy as np
from argparse import ArgumentParser
import time
from models.face_detection import Face_Detection
from models.facial_landmarks import Facial_Landmark
from models.gaze_estimation import Gaze_Estimation
from models.head_pose_estimation import Head_Pose_Estimation
from mouse_controller import MouseController
from input_feeder import InputFeeder

face_detection_name = "../resources/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
gaze_estimation_name = "../resources/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"
head_pose_estimation_name = "../resources/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"
facial_landmarks_name = "../resources/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009"
video = "../bin/demo.avi"
image = "../bin/staring.jpg"


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument('--media_file', type=str, required=False,
                        default=video, help="Specify the path to the file")
    parser.add_argument('--media_type', type=str, required=False,
                        default="video", help="Specify the type of file: video or cam")
    parser.add_argument('--speed', required=False, type=str, default="fast",
                        help="Specify the speed of the cursor: slow, medium or fast"
                            "(fast by default)")
    parser.add_argument('--precision', required=False, type=str, default="high",
                        help="Specify the precision of the mouse: low, medium or high"
                            "(high by default)")
    parser.add_argument('--get_perf_counts', required=False, type=str, default="false",
                        help="Benchmark the running times of different parts"
                            "of the preprocessing and inference pipeline"
                            "Specify if active or not: true or false (false by default)")
    parser.add_argument('--iterations', required=False, default=1,
                        help="Number of iterations on the video"
                             "Specify the number of iterations done in the video, for test purposes (10 by default)")
    parser.add_argument('--batch_size', required=False, type=int, default=10,
                        help="Number of frames skipped between inferences"
                             "Specify the number of frames to be skipped between inferences (10 by default)")
    parser.add_argument('--show_video', required=False, default="True",
                        help="Indicate if the video is shown or not"
                             "Specify if True or False (True by default)")

    return parser

def main(args):
    if args.get_perf_counts.lower() == "true":
        perf_counts = True
    elif args.get_perf_counts.lower() == "false":
        perf_counts = False

    precision = args.precision.lower()
    speed = args.speed.lower()
    media_type = args.media_type.lower()
    media_path = args.media_file
    toggle_UI = False if args.show_video.lower() == "false" else True
    batch_size = args.batch_size
    #initialize the mouse object
    mouse = MouseController(precision, speed)

    # Initialize the input feeder
    feed = InputFeeder(media_type, batch_size, media_path)


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

    for _ in range(int(args.iterations)):
        feed.load_data()
        times = np.zeros((8, ))
        counter_frames = 0
        if media_type != "image":
            width = feed.cap.get(3)
            height = feed.cap.get(4)
        else:
            height, width, _ = feed.cap.shape
        try:
            for frame in feed.next_batch(media_type):
                counter_frames += 1
                # Process the input image through the face detection first
                # So we can ignore anything that isn't the head
                start_time = time.time()
                p_image_f = face_detection.preprocess_input(frame)
                times[0] += time.time() - start_time
                start_time = time.time()
                output_f = face_detection.predict(p_image_f)
                times[1] += time.time() - start_time
                f_image = face_detection.preprocess_output(frame, output_f, width, height)
                f_height, f_width, _ = f_image.shape

                # Then we process the output through the head pose estimation
                start_time = time.time()
                p_image_h = head_pose.preprocess_input(f_image)
                times[2] += time.time() - start_time
                start_time = time.time()
                output_h = head_pose.predict(p_image_h)
                times[3] += time.time() - start_time
                head_position = head_pose.preprocess_output(output_h)

                # And we get the coordinates of the eyes
                start_time = time.time()
                p_image_l = facial_landmarks.preprocess_input(f_image)
                times[4] += time.time() - start_time
                start_time = time.time()
                output_l = facial_landmarks.predict(p_image_l)
                times[5] += time.time() - start_time
                left_eye_frame, right_eye_frame = facial_landmarks.preprocess_output(output_l, f_image, f_width, f_height)

                # We use the coordinates and the face detection to get an estimate of the gaze
                start_time = time.time()
                left_eye = gaze_estimation.preprocess_input(left_eye_frame)
                right_eye = gaze_estimation.preprocess_input(right_eye_frame)
                times[6] += time.time() - start_time
                start_time = time.time()
                output_g = gaze_estimation.predict(left_eye, right_eye, head_position)
                times[7] += time.time() - start_time

                x, y, gaze_vector = gaze_estimation.preprocess_output(output_g)
                mouse.move(x,y)
                if perf_counts:
                    cv2.putText( frame, "Preprocess Face Detection: " + str(times[0]/counter_frames* 1000) + " ms", (0,50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0), 3)
                    cv2.putText(frame, "Inference Face Detection: " + str(times[1]/counter_frames * 1000) + " ms", (0, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0), 3)
                    cv2.putText(frame, "Preprocess Facial Landmarks: " + str(times[2]/counter_frames* 1000) + " ms", (0, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0), 3)
                    cv2.putText(frame, "Inference Facial Landmarks: " + str(times[3]/counter_frames* 1000) + " ms", (0, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0), 3)
                    cv2.putText(frame, "Preprocess Head Pose: " + str(times[4]/counter_frames* 1000) + " ms", (0, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0), 3)
                    cv2.putText(frame, "Inference Head Pose: " + str(times[5]/counter_frames* 1000) + " ms", (0, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0), 3)
                    cv2.putText(frame, "Preprocess Gaze Estimation: " + str(times[6]/counter_frames* 1000) + " ms", (0, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0), 3)
                    cv2.putText(frame, "Inference Gaze Estimation: " + str(times[7]/counter_frames* 1000) + " ms", (0, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0), 3)
                    print("Preprocess Face Detection: " + str(times[0] / counter_frames * 1000) + " ms")
                    print("Inference Face Detection: " + str(times[1] / counter_frames * 1000) + " ms")
                    print("Preprocess Facial Landmarks: " + str(times[2] / counter_frames * 1000) + " ms")
                    print("Inference Facial Landmarks: " + str(times[3] / counter_frames * 1000) + " ms")
                    print("Preprocess Head Pose: " + str(times[4] / counter_frames * 1000) + " ms")
                    print("Inference Head Pose: " + str(times[5] / counter_frames * 1000) + " ms")
                    print("Preprocess Gaze Estimation: " + str(times[6] / counter_frames * 1000) + " ms")
                    print("Inference Gaze Estimation: " + str(times[7] / counter_frames * 1000) + " ms")
                    if toggle_UI:
                        cv2.imshow("Frame", frame)
                else:
                    cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.waitKey(1) & 0xFF == ord('i'):
                    toggle_UI = False if toggle_UI else True

        except:
            print("Video has ended or couldn't continue")
        if perf_counts:
            print("Final average: ")
            print("Preprocess Face Detection: " + str(times[0] / counter_frames * 1000) + " ms")
            print("Inference Face Detection: " + str(times[1] / counter_frames * 1000) + " ms")
            print("Preprocess Facial Landmarks: " + str(times[2] / counter_frames * 1000) + " ms")
            print("Inference Facial Landmarks: " + str(times[3] / counter_frames * 1000) + " ms")
            print("Preprocess Head Pose: " + str(times[4] / counter_frames * 1000) + " ms")
            print("Inference Head Pose: " + str(times[5] / counter_frames * 1000) + " ms")
            print("Preprocess Gaze Estimation: " + str(times[6] / counter_frames * 1000) + " ms")
            print("Inference Gaze Estimation: " + str(times[7] / counter_frames * 1000) + " ms")
        feed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
