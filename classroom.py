import argparse
import cv2
import numpy as np
from inference import Network
from openvino.inference_engine import IENetwork, IECore
import pylab as plt
import math
import matplotlib
from scipy.ndimage.filters import gaussian_filter

INPUT_STREAM  = "classroom.mp4"
CPU_EXTENSION = "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll"
MODEL         = "models\person-detection-action-recognition-0006.xml"
# if linux : /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# Constants
THRE1 = 0.3

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    t_desc = "The confidence thresholds used to draw bounding boxes"
    ###       2) The user choosing the color of the bounding boxes
    c_desc = "The color name of the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-t", help=t_desc, default=0.2)
    optional.add_argument("-c", help=c_desc, default="green")
    args = parser.parse_args()

    return args

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    #image = image.reshape(1, 3, height, width)
    #print("in preprocessing", *image.shape) # same thine : in preprocessing 3 384 672
    image = image.reshape(1, *image.shape)

    return image

def infer_on_video(args):
    '''
    Performs inference on video - main method
    '''
    ### Load the network model into the IE
    print("Load the network model into the IE")
    net = Network()
    net.load_model(MODEL, "CPU")
    #net.load_model(MODEL, "CPU", CPU_EXTENSION)

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out-' + INPUT_STREAM, 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    frame_count = 0;
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        
        key_pressed = cv2.waitKey(60)
        preprocessed_frame = preprocessing(frame, net.get_input_shape()[2], net.get_input_shape()[3])
        #print("Perform inference on the frame")
        net.async_inference(preprocessed_frame)

        if net.wait() == 0:
            # Get the output of inference
            output_blobs = net.extract_output()
            print("ouput.shape:",output_blobs['ActionNet/action_heads/out_head_2_anchor_1'].shape)
            print("ouput[0][0]:",output_blobs['ActionNet/action_heads/out_head_2_anchor_1'][0][0])

        # Write a frame here for debug purpose
        cv2.imwrite("classroom-frame" + str(frame_count) + ".png", frame)
     
        # Write out the frame in the video
        out.write(frame)

        # frame count
        frame_count = frame_count + 1
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Starting")
    args = get_args()
    infer_on_video(args)

if __name__ == "__main__":
    main()
