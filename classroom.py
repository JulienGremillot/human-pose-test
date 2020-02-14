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
MODEL         = "C:\\Users\\gremi\\Documents\\Julien\\udacity_intel\\models\\intel\\person-detection-action-recognition-0005\\FP16\\person-detection-action-recognition-0005.xml"

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

ACTIONS = ['sitting', 'standing', 'raising hand']

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
    
def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask

def create_output_image(image, output):
    '''
    creates an output image showing the result of inference.
    '''
    # Remove final part of output not used for heatmaps
    output = output[:-1]
    # Get only pose detections above 0.5 confidence, set to 255
    #for c in range(len(output)):
    #    output[c] = np.where(output[c]>0.5, 255, 0)
    # Sum along the "class" axis
    output = np.sum(output, axis=0)
    # Get semantic mask
    pose_mask = get_mask(output)
    # Combine with original image
    image = image + pose_mask
    #return image.astype('uint8')
    return pose_mask.astype('uint8')

def infer_on_video(args):
    '''
    Performs inference on video - main method
    '''
    ### Load the network model into the IE
    print("Load the network model into the IE")
    net = Network()
    net.load_model(MODEL, "CPU", CPU_EXTENSION)

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
            output = net.extract_output()
            #print("output:", output.keys())
            #print("output['mbox_loc1/out/conv/flat']:", output['mbox_loc1/out/conv/flat'][0].shape)
            #print("output['mbox_main_conf/out/conv/flat/softmax/flat']:", output['mbox_main_conf/out/conv/flat/softmax/flat'][0])

            # trying to use the 4 anchors
            for anchor in range(0, 4):
                anchor_confidences = np.reshape(output['out/anchor'+str(anchor+1)][0], (3, 25, 43))
                for i in range(0, 3):
                    for j in range(0, 25):
                        for k in range(0, 43):
                            if anchor_confidences[i][j][k] > 0.5:
                                print("anchor_confidences[", i, "][", j, "][", k, "]:", anchor_confidences[i][j][k])
                                frame = cv2.putText(frame,
                                                   "Anchor{}: {}".format(anchor+1, ACTIONS[i]),
                                                   (50, 50 * (anchor + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                                #print('output[mbox_loc1/out/conv/flat].shape=', output['mbox_loc1/out/conv/flat'].shape)
                                #boxes_coords = np.reshape(output['mbox_loc1/out/conv/flat'][0], (4, 25, 43, 4))
                                boxes_coords = np.reshape(output['mbox/priorbox'][0][0], (4, 25, 43, 4))
                                #print('boxes_coords[0][j][k]=', boxes_coords[0][j][k])
                                #print('boxes_coords[1][j][k]=', boxes_coords[1][j][k])
                                #print('boxes_coords[2][j][k]=', boxes_coords[2][j][k])
                                print('boxes_coords['+str(anchor)+'][j][k]=', boxes_coords[anchor][j][k])
                                y_min = int(boxes_coords[anchor][j][k][0] * width)
                                x_min = int(boxes_coords[anchor][j][k][1] * height)
                                y_max = int(boxes_coords[anchor][j][k][2] * width)
                                x_max = int(boxes_coords[anchor][j][k][3] * height)
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
                                #cv2.imwrite("frame" + str(frame_count) + ".png", frame)
            #print('num_priors:', num_priors[0][0][0])
            #probs = output['prob_emotion'][0]
            #index_of_maximum = np.argmax(probs)
            #emotion = EMOTIONS[index_of_maximum]
            #if index_of_maximum == 0:
            #    probs[0] = 0
            #    emotion = emotion + " (" + EMOTIONS[np.argmax(probs)] + ")"
            #print("emotion=", emotion)

            # Scale the output text by the image shape
            #scaler = max(int(frame.shape[0] / 1000), 1)

            # Write the text of color and type onto the image
            #frame = cv2.putText(frame,
             #                   "Detected: {}".format(emotion),
             #                   (750 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
             #                   scaler, (0, 0, 0), 3 * scaler)

        # Write a frame here for debug purpose
        #cv2.imwrite("frame" + str(frame_count) + ".png", frame)
     
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
