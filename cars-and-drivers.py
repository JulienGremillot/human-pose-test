import argparse
import cv2
import numpy as np
from inference import Network
# git clone https://github.com/vaab/colour
from colour import Color
from openvino.inference_engine import IENetwork, IECore

INPUT_STREAM = "bmw2.mp4"
CPU_EXTENSION = "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll"
#"C:\Users\gremi\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release\cpu_extension.lib"
#/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

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

def create_output_image(image, output, width, height, color, confidence):
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
        if conf >= confidence:
            x_min = int(bounding_box[3] * width)
            y_min = int(bounding_box[4] * height)
            x_max = int(bounding_box[5] * width)
            y_max = int(bounding_box[6] * height)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

    
def infer_on_video(args):
    ### Initialize the Inference Engine
    #ie = IECore()

    ### Load the face network model into the IE
    facenet = Network()
    facenet.load_model("C:\\Users\\gremi\\Documents\\Julien\\udacity_intel\\models\\face-detection-adas-0001\\FP16\\face-detection-adas-0001.xml", "CPU", CPU_EXTENSION)
    
    carnet = Network()
    carnet.load_model("C:\\Users\\gremi\\Documents\\Julien\\udacity_intel\\models\\vehicle-detection-adas-0002\\FP16\\vehicle-detection-adas-0002.xml", "CPU", CPU_EXTENSION)

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
    
    color = [int(c*255) for c in Color(args.c).rgb]
    #print("Color=", color)
    
    # Process frames until the video ends, or process is exited
    car_count = 0;
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### Pre-process the frame
        #print("Preprocessing frame", width, "x", height, 
        #      "to network dimensions", net.get_input_shape()[2], "x", net.get_input_shape()[3])
        preprocessed_frame = preprocessing(frame, 
                                           carnet.get_input_shape()[2], 
                                           carnet.get_input_shape()[3])

        ### Perform car inference on the frame
        #print("Perform inference on the frame")
        carnet.async_inference(preprocessed_frame)

        ### Get the output of inference
        if carnet.wait() == 0:
            #print("Get the output of inference")
            caroutput = carnet.extract_output()
            #print("OK, output=", output)

            # get cropped car image
            for bounding_box in caroutput[0][0]:
                conf = bounding_box[2]
                if conf >= float(args.t):
                    x = int(bounding_box[3] * width)
                    if x < 0:
                        x = 0
                    y = int(bounding_box[4] * height)
                    if y < 0:
                        y = 0
                    w = int(bounding_box[5] * width) - x
                    h = int(bounding_box[6] * height) - y
                    # getRectSubPix(InputArray image, Size patchSize, Point2f center)
                    carimg = cv2.getRectSubPix(frame, (w, h), (x + w/2, y + h/2))
                    #print("car found #", car_count, ", coords=", (x + w/2, y + h/2), ", w/h=", (w, h), "shape=", carimg.shape)
                    #print("car found at ", x, ",", y, ", dims=",w,"x",h,", shape=", carimg.shape)
                    preprocessed_carimg = preprocessing(carimg, 
                                           facenet.get_input_shape()[2], 
                                           facenet.get_input_shape()[3])
                    
                    # perform face inference in this cropped car image
                    facenet.async_inference(preprocessed_carimg)
                    
                    if facenet.wait() == 0:
                        faceoutput = facenet.extract_output()
                        if len(faceoutput[0][0]) > 0:
                            face_bounding_box = faceoutput[0][0][0] # take only the first face
                            conf = face_bounding_box[2]
                            if conf >= float(args.t):
                                x_face = int(face_bounding_box[3] * carimg.shape[1])
                                y_face = int(face_bounding_box[4] * carimg.shape[0])
                                w_face = int(face_bounding_box[5] * carimg.shape[1]) - x_face
                                h_face = int(face_bounding_box[6] * carimg.shape[0]) - y_face
                                #print("face found - x_face=",x_face," y_face=", y_face, " w_face=",w_face, " h_face=",h_face)
                                faceimg = cv2.getRectSubPix(carimg, (w_face, h_face), (x_face + w_face/2, y_face + h_face/2))
                                # double the size of the face
                                faceimg = cv2.resize(faceimg,(int(w_face * 2),int(h_face * 2)))
                                #cv2.imwrite('out-face-' + INPUT_STREAM + str(car_count) + '.png', faceimg)
                                cv2.rectangle(faceimg, (0, 0), (faceimg.shape[1], faceimg.shape[0]), (0, 255, 0), 2)
                                #print("face found on car ", car_count, ", coords=", (x_face + w_face/2, y_face + h_face/2), ", w/h=", (w_face, h_face), "shape=", faceimg.shape)
                                #car_count += 1
                                #cv2.imwrite('out-' + INPUT_STREAM + str(car_count) + '.png', carimg)

                                # paste the detected face in the corner of the car bounding box
                                #print("paste face to ",y,":",y+faceimg.shape[0],", ",x,":",x+faceimg.shape[1])
                                #cv2.imwrite('out-' + INPUT_STREAM + str(faces_count) + '.png', faceimg)
                                #faces_count += 1
                                frame[y+1:y+faceimg.shape[0]+1, x+1:x+faceimg.shape[1]+1] = faceimg

                            ### Update the frame to include detected cars bounding boxes
                            frame = create_output_image(frame, caroutput, width, height, (0, 0, 255), float(args.t))

                            #frame = create_output_image(frame, faceoutput, width, height, (0, 255, 0), float(args.t))

                            #blob = output['Mconv7_stage2_L2']
                            # Resize the heatmap back to the size of the input
                            #heatmap = np.zeros([blob.shape[1], input_shape[0], input_shape[1]])
                            #for h in range(len(blob[0])):
                            #    heatmap[h] = cv2.resize(blob[0][h], (input_shape[1], input_shape[0]))

                        # Write out the frame
                        out.write(frame)
            
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
