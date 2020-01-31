import argparse
import cv2
import numpy as np
from inference import Network
from openvino.inference_engine import IENetwork, IECore

import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

MODEL         = "emotions-recognition-retail-0003.xml"
# if linux : /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'anger']

INPUT_WIDTH = 640
INPUT_HEIGHT = 480

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
    optional.add_argument("-i", help=i_desc, default='')
    optional.add_argument("-d", help=d_desc, default='MYRIAD')
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
    net.load_model(MODEL, 'MYRIAD')

    print("[INFO] starting video stream...")
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()

    # capture frames from the camera
    # loop over the frames from the video stream
    frame_count = 0;
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=INPUT_WIDTH)

        key_pressed = cv2.waitKey(60)

        preprocessed_frame = preprocessing(frame, net.get_input_shape()[2], net.get_input_shape()[3])
        #print("Perform inference on the frame")
        net.async_inference(preprocessed_frame)

        if net.wait() == 0:
            # Get the output of inference
            output_blobs = net.extract_output()
            probs = output_blobs['prob_emotion'][0]
            index_of_maximum = np.argmax(probs)
            emotion = EMOTIONS[index_of_maximum]
            if index_of_maximum == 0:
                probs[0] = 0
                emotion = emotion + " (" + EMOTIONS[np.argmax(probs)] + ")"
            print("emotion=", emotion)

            # Scale the output text by the image shape
            scaler = max(int(frame.shape[0] / 1000), 1)

            # Write the text of color and type onto the image
            frame = cv2.putText(frame,
                                "Detected: {}".format(emotion),
                                (50, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                                scaler, (255, 255, 255), 3 * scaler)

        # Write a frame here for debug purpose
        #cv2.imwrite("frame" + str(frame_count) + ".png", frame)

        cv2.imshow("window name", frame)

        # frame count
        frame_count = frame_count + 1
        # update the FPS counter
        fps.update()

        # Break if escape key pressed
        if key_pressed == 27:
            break

    print("stop capture!")
    # Release the out writer, capture, and destroy any OpenCV windows
    cv2.destroyAllWindows()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    vs.stop()

def main():
    print("Starting")
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
