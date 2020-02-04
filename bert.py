import argparse
import numpy as np
from openvino.inference_engine import IENetwork, IECore
from input_feature import get_input_feature

CPU_EXTENSION = "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll"
MODEL_XML = "C:\\Users\\gremi\\Documents\\Julien\\multilingual_L-12_H-768_A-12\\bert_model.ckpt.xml"
MODEL_BIN = "C:\\Users\\gremi\\Documents\\Julien\\multilingual_L-12_H-768_A-12\\bert_model.ckpt.bin"

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
    optional.add_argument("-i", help=i_desc, default='')
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-t", help=t_desc, default=0.2)
    optional.add_argument("-c", help=c_desc, default="green")
    args = parser.parse_args()

    return args

def infer(args):
    '''
    Performs inference on video - main method
    '''
    ie = IECore()
    ie.add_extension(CPU_EXTENSION, 'CPU')

    ### Load the network model into the IE
    print("Load the network model into the IE")
    net = IENetwork(model=MODEL_XML, weights=MODEL_BIN)
    net.batch_size = 1

    # prepare input and output
    input_blob = [i for i in net.inputs]
    out_blob = [i for i in net.outputs]

    print('input:', input_blob)

    input_ids = 'Placeholder'
    input_mask = 'Placeholder_1'
    input_segment_ids = 'Placeholder_2'

    print('output:', out_blob)

    sentences = ["Hello world!"]
    feature = get_input_feature(sentences)
    input_ids_blob = np.array(feature[0].input_ids).reshape((1, 128))
    input_mask_blob = np.array(feature[0].input_mask).reshape((1, 128))
    input_segment_ids_blob = np.array(feature[0].segment_ids).reshape((1, 128))

    in1 = np.ones([1, 128]).astype(np.int32)
    in2 = np.ones([1, 128]).astype(np.int32)
    in3 = np.ones([1, 128]).astype(np.int32)

    exec_net = ie.load_network(network=net, device_name="CPU")
    res = exec_net.infer(inputs={input_blob[0]: in1, input_blob[1]: in2, input_blob[2]: in3})
    #res = res[out_blob[0]]
    print("res[", out_blob[0], "].shape = ", res[out_blob[0]].shape)
    print("res[", out_blob[1], "].shape = ", res[out_blob[1]].shape)

def main():
    print("Starting")
    args = get_args()
    infer(args)

if __name__ == "__main__":
    main()
