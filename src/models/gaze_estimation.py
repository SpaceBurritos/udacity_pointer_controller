'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore
import sys
import cv2
import numpy as np
import math

class Gaze_Estimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''

        '''
        self.device = device
        self.model_weights = model_name + ".bin"
        self.model_structure = model_name + ".xml"
        self.extensions = extensions
        self.core = IECore()
        try:
            try:
                self.model = self.core.read_network(model=self.model_structure, weights=self.model_weights)
            except:
                self.model = IENetwork(self.model_structure, self.model_weights)
        except:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path?")
        # raise NotImplementedError

        self.input_name = [x for x in self.model.inputs]
        self.input_shape = self.model.inputs[self.input_name[0]].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''

        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Usupported Layers!")
            sys.exit(1)

    def predict(self, left_eye, right_eye, head_pose):
        '''

        This method is meant for running predictions on the input image.
        '''
        images = [head_pose, left_eye, right_eye]
        input_dict = {i:e for i,e in zip(self.input_name, images)}
        infer_request = self.network.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request.wait()
        if infer_status == 0:
            output = infer_request.outputs

        return output[self.output_name][0]#np.array([output_head["gaze_vector"][0], output_left["gaze_vector"][0], output_right["gaze_vector"][0]])


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_image = cv2.resize(image, (60, 60))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1,*p_image.shape)
        return p_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        gaze_vector = outputs
        roll = gaze_vector[2]
        gaze_vector = gaze_vector/np.linalg.norm(gaze_vector)
        cs = math.cos(roll * math.pi/180)
        sn = math.sin(roll*math.pi/180)

        x = gaze_vector[0] * cs + gaze_vector[1] * sn
        y = gaze_vector[0] * sn + gaze_vector[1] * cs

        return x, y, gaze_vector
