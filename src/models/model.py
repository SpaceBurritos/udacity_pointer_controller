'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore
import sys
import cv2
from .face_detection import Face_Detection
from .facial_landmarks import Facial_Landmark
from .gaze_estimation import Gaze_Estimation
from .head_pose_estimation import Head_Pose_Estimation

class Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, face_estimation_path, facial_landmarks_path, gaze_estimation_path,
                 head_pose_estimation_path, device='CPU', extensions=None):
        '''

        '''
        # Initialize the gaze estimation model
        self.gaze_estimation = Gaze_Estimation(gaze_estimation_path, device)

        # Initialize the head pose estimation model
        self.head_pose = Head_Pose_Estimation(head_pose_estimation_path, device)

        # Initialize the facial landmarks model
        self.facial_landmarks = Facial_Landmark(facial_landmarks_path, device)

        # Initialize the face detection model
        self.face_detection = Face_Detection(face_estimation_path, device)

    def load_models(self):
        '''
        This method is for loading the model to the device specified by the user.
        '''
        # load the gaze estimation model
        self.gaze_estimation.load_model()

        # load the head pose estimation model
        self.head_pose.load_model()

        # load the facial landmarks model
        self.facial_landmarks.load_model()

        # load model the face detection model
        self.face_detection.load_model()

    def predict(self, frame, width, height, times):
        '''

        This method is meant for running predictions on the input image.
        '''

        # Process the input image through the face detection first
        # So we can ignore anything that isn't the head
        f_image, f_height, f_width, times = self.face_detection.predict(frame, width, height, times)

        # Then we process the output through the head pose estimation
        head_position, times = self.head_pose.predict(f_image, times)

        # And we get the coordinates of the eyes
        left_eye_frame, right_eye_frame, times = self.facial_landmarks.predict(f_image, f_width, f_height, times)

        # We use the coordinates and the face detection to get an estimate of the gaze
        x, y, gaze_vector, times = self.gaze_estimation.predict(left_eye_frame, right_eye_frame, head_position, times)
        return x, y, gaze_vector, times
