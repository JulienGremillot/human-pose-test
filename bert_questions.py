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

    exec_net = ie.load_network(network=net, device_name="CPU")

    out_blob = [i for i in net.outputs]

    input_ids = 'Placeholder'
    input_mask = 'Placeholder_1'
    input_segment_ids = 'Placeholder_2'

    # load questions from txt file
    with open('questions.txt') as f:
        questions = [v.strip() for v in f]
        print('questions number:{}'.format(len(questions)))

    feature = get_input_feature(questions)

    vectors = []
    for i in range(len(feature)):
        input_ids_blob = np.array(feature[i].input_ids).reshape((1, 128))
        input_mask_blob = np.array(feature[i].input_mask).reshape((1, 128))
        input_segment_ids_blob = np.array(feature[i].segment_ids).reshape((1, 128))
        res = exec_net.infer(
            inputs={input_ids: input_ids_blob, input_mask: input_mask_blob, input_segment_ids: input_segment_ids_blob})
        vectors.append(res[out_blob[1]])

    vectors = np.array(vectors)
    vectors = vectors.reshape((vectors.shape[0], vectors.shape[-1]))

    # type question and search it with 5 most similar in stored question file.
    topk = 5
    while True:
        query_sentence = input('your questions: ')
        query_list = []
        query_list.append(query_sentence.strip())
        feature = get_input_feature(query_list)
        input_ids_blob = np.array(feature[0].input_ids).reshape((1, 128))
        input_mask_blob = np.array(feature[0].input_mask).reshape((1, 128))
        input_segment_ids_blob = np.array(feature[0].segment_ids).reshape((1, 128))
        res = exec_net.infer(inputs={input_ids: input_ids_blob, input_mask: input_mask_blob, input_segment_ids: input_segment_ids_blob})
        query_vec = res[out_blob[1]]

        # compute normalized dot product as score
        score = np.sum(query_vec * vectors, axis=1) / np.linalg.norm(vectors, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        print('top %d questions similar to "%s"' % (topk, query_sentence))
        for idx in topk_idx:
            print('> %s\t%s' % ('%.1f' % score[idx], questions[idx]))

def main():
    print("Starting")
    args = get_args()
    infer(args)

if __name__ == "__main__":
    main()
