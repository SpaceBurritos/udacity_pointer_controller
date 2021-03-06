import cv2
import numpy as np
from argparse import ArgumentParser
from mouse_controller import MouseController
from input_feeder import InputFeeder
from models.model import Model

face_detection = "../resources/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
gaze_estimation = "../resources/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"
head_pose_estimation = "../resources/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"
facial_landmarks = "../resources/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009"

def build_argparser():
    """
    Parse command line arguments.
    """
    parser = ArgumentParser()

    parser.add_argument('--media_file', type=str, required=True,
                        help="Specify the path to the file (type none if a webcam will be used")
    parser.add_argument('--media_type', type=str, required=True,
                        help="Specify the type of file: video or cam")
    parser.add_argument('--speed', required=False, type=str, default="fast",
                        help="Specify the speed of the cursor: slow, medium or fast"
                            "(fast by default)")
    parser.add_argument('--precision', required=False, type=str, default="high",
                        help="Specify the precision of the mouse: low, medium or high"
                            "(high by default)")
    parser.add_argument('--device', required=False, type=str, default="CPU",
                        help="Specify the hardware"
                            "(CPU by default)")
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
    parser.add_argument('--show_video', required=False, default=True,
                        help="Indicate if the video is shown or not"
                             "Specify if True or False (True by default)")

    return parser

def main(args):


    # getting the arguments
    if args.get_perf_counts.lower() == "true":
        perf_counts = True
    elif args.get_perf_counts.lower() == "false":
        perf_counts = False
    precision = args.precision.lower()
    speed = args.speed.lower()
    media_type = args.media_type.lower()
    media_path = args.media_file
    toggle_ui = args.show_video
    print(toggle_ui)
    batch_size = args.batch_size
    device = args.device
    iterations = 1 if media_type == "cam" else int(args.iterations)


    #initialize the mouse object
    mouse = MouseController(precision, speed)

    # Initialize the input feeder
    feed = InputFeeder(media_type, batch_size, media_path)

    # Initialize and load the inference models
    model = Model(face_detection, facial_landmarks, gaze_estimation, head_pose_estimation, device)
    model.load_models()

    for _ in range(iterations):

        feed.load_data()

        #This will be used as a way to keep track of the average time for the preprocessing and inference of the models
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
                #generates the prediction
                x,y, gaze_vector, times = model.predict(frame, width, height, times)
                #generates the movement on the cursor
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

                if toggle_ui == True:
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
