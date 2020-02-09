import argparse
import os
import cv2
from openvino.inference_engine import IENetwork, IECore

INPUT_STREAM  = "vids\\vid4.mp4"
CPU_EXTENSION = "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll"

# To keep track of this model origin :
# - I downloaded it from http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
# - I converted it with :
#   python mo_tf.py --input_meta_graph model.ckpt.meta (...) => but it was bad !
# - I checked with :
#   python "C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\samples\python_samples\object_detection_sample_ssd\object_detection_sample_ssd.py"
#     -i C:\Users\gremi\test.jpg
#     -m C:\Users\gremi\PycharmProjects\human-pose-test\frozen_inference_graph.xml
#     -d CPU --cpu_extension "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"
# - Then I retried with the frozen model :
#   python mo_tf.py --input_model C:\Users\gremi\PycharmProjects\human-pose-test\ssdlite_mobilenet_v2_coco\frozen_inference_graph.pb
#     --tensorflow_use_custom_operations_config extensions\front\tf\ssd_v2_support.json
#     --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco.config
#     --data_type FP16

MODEL  = "frozen_inference_graph.xml"
# This is from https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
# I converted it to a simple array, with index corresponding to the index used by the model (some missing indexes)
LABELS = [ "", "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",\
           "fire hydrant","", "stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",\
           "bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis","snowboard",\
           "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","",\
           "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",\
           "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","","dining table","","","toilet","","tv","laptop",\
           "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","","book","clock",\
           "vase","scissors","teddy bear","hair drier","toothbrush" ]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# confidence for detected bounding boxes
CONFIDENCE = 0.8

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"

    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
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

def add_text_to_bounding_box(image, text, x_min, y_min, color):
    '''
    Add the text label to the bounding box
    '''
    labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image, (x_min, y_min), (x_min + labelSize[0][0] + 10, y_min - labelSize[0][1] - 10), color, cv2.FILLED)
    image = cv2.putText(image, text, (x_min + 5, y_min - labelSize[0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return image

def create_output_image(image, output, width, height, color):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    #The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected
        # bounding boxes. For each detection, the description has the format:
        # [image_id, label, conf, x_min, y_min, x_max, y_max]
        #image_id - ID of the image in the batch
        #label - predicted class ID
        #conf - confidence for the predicted class
        #(x_min, y_min) - coordinates of the top left bounding box corner
        #(x_max, y_max) - coordinates of the bottom right bounding box corner.
    thickness = 1 # in pixels
    for bounding_box in output[0][0]:
        conf = bounding_box[2]
        if conf >= CONFIDENCE:
            x_min = int(bounding_box[3] * width)
            y_min = int(bounding_box[4] * height)
            x_max = int(bounding_box[5] * width)
            y_max = int(bounding_box[6] * height)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
            image = add_text_to_bounding_box(image, LABELS[int(bounding_box[1])], x_min, y_min, color)
    return image

def infer_on_video(args):
    '''
    Performs inference on video - main method
    '''
    ### Load the network model into the IE
    print("Load the network model into the IE")
    plugin = IECore()
    plugin.add_extension(CPU_EXTENSION, args.d)
    network = IENetwork(model=MODEL, weights=os.path.splitext(MODEL)[0] + ".bin")
    exec_network = plugin.load_network(network, args.d)
    input_blob = next(iter(network.inputs))
    input_shape = network.inputs[input_blob].shape
    #output_blob = next(iter(network.outputs))

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out-cat.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    frame_count = 0;
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        
        key_pressed = cv2.waitKey(60)
        preprocessed_frame = preprocessing(frame, input_shape[2], input_shape[3])
        #print("Perform inference on the frame")
        exec_network.start_async(request_id=0, inputs={input_blob: preprocessed_frame})

        if exec_network.requests[0].wait(-1) == 0:
            # Get the output of inference
            output = exec_network.requests[0].outputs
            frame = create_output_image(frame, output['DetectionOutput'], width, height, (0, 0, 255))

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
