'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore
import sys
import cv2

class Head_Pose_Estimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
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

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Usupported Layers!")
            sys.exit(1)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        infer_request = self.network.start_async(request_id=0, inputs={self.input_name: image})
        infer_status = infer_request.wait()
        if infer_status == 0:
            output = infer_request.outputs
        return output


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1,*p_image.shape)
        return p_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        yaw = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]
        return [yaw, pitch, roll]
