'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        return


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        #while True:
        #    status = self.exec_network.requests[0].wait(-1)
        #    if status == 0:
        #        break
        #    else:
        #        time.sleep(1)
        status = self.exec_network.requests[0].wait(-1)
        return status


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        #The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected
        # bounding boxes. For each detection, the description has the format: 
        # [image_id, label, conf, x_min, y_min, x_max, y_max]
        #image_id - ID of the image in the batch
        #label - predicted class ID
        #conf - confidence for the predicted class
        #(x_min, y_min) - coordinates of the top left bounding box corner
        #(x_max, y_max) - coordinates of the bottom right bounding box corner.
        #detection_out = self.network.outputs['detection_out']
        # we'll got np.zeros([N, 7])
        #boxes = np.zeros([detection_out.shape[2], detection_out.shape[3]])
        #for b in range(detection_out.shape[2]):
        #    print(detection_out[0][0][b])
            #boxes[b] = cv2.resize(detection_out[0][0][b], (input_shape[1], input_shape[0]))
        #print(n)
        #print(detection_out[::,1])
        #reshaped = self.exec_network.reshape({detection_out: (one, two, n, seven)})
        #print(reshaped)
        #return self.exec_network.requests[0].outputs[self.output_blob]
        return self.exec_network.requests[0].outputs
