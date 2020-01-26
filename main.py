import argparse
import cv2
import numpy as np
from inference import Network
from openvino.inference_engine import IENetwork, IECore
import pylab as plt
import time
import math
import matplotlib
import scipy
from scipy.ndimage.filters import gaussian_filter

INPUT_STREAM  = "kungfu.mp4"
CPU_EXTENSION = "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll"
MODEL         = "C:\\Users\\gremi\\Documents\\Julien\\udacity_intel\\lesson4\\models\\human-pose-estimation-0001.xml"
# if linux : /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# find connection in the specified sequence, center 29 is in the position 15
LIMB_SEQ = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

# the middle joints heatmap correpondence
MAP_IDX = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

# Constants
THRE1 = 0.1
THRE2 = 0.05
MID_NUM = 10
STRIDE = 8
PAD_VALUE = 128

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

def padRightDownCorner(img):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%STRIDE==0) else STRIDE - (h % STRIDE) # down
    pad[3] = 0 if (w%STRIDE==0) else STRIDE - (w % STRIDE) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + PAD_VALUE, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + PAD_VALUE, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + PAD_VALUE, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + PAD_VALUE, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

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

def handle_pose(blob, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # Resize the heatmap back to the size of the input
    heatmap = np.zeros([blob.shape[1], input_shape[0], input_shape[1]])
    print("blob[0][0].shape=", blob[0][0].shape, " blob[0][0]=",blob[0][0])
    for h in range(len(blob[0])):
        heatmap[h] = cv2.resize(blob[0][h], (input_shape[1], input_shape[0]))
    print("heatmap.shape=", heatmap.shape)
    return heatmap
    
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

def adding_sticks_to_frame(frame, subset, candidate):
    '''
    Adds sticks (lines) between the dots.
    Return the modified frame.
    '''
    stickwidth = 4
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(LIMB_SEQ[i])-1]
            if -1 in index:
                continue
            cur_canvas = frame.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, COLORS[i])
            frame = cv2.addWeighted(frame, 0.4, cur_canvas, 0.6, 0)
    return frame

def compute_peaks(heatmap):
    '''
    Compute an array of peaks
    '''
    all_peaks = []
    peak_counter = 0
    for part in range(19 - 1):
        map_ori = heatmap[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > THRE1))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks_with_score))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks_with_score)
    return all_peaks


def search_connection_candidate(candA, candB, nA, nB, oriImg, score_mid):
    '''
    Search the connection candidate
    '''
    connection_candidate = []
    for i in range(nA):
        for j in range(nB):
            vec = np.subtract(candB[j][:2], candA[i][:2])
            norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
            vec = np.divide(vec, norm)

            startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=MID_NUM), \
                                np.linspace(candA[i][1], candB[j][1], num=MID_NUM)))

            vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                              for I in range(len(startend))])
            vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                              for I in range(len(startend))])

            score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])

            try:
                score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
            except ZeroDivisionError:
                score_with_dist_prior = -1

            criterion1 = len(np.nonzero(score_midpts > THRE2)[0]) > 0.8 * len(score_midpts)
            criterion2 = score_with_dist_prior > 0
            if criterion1 and criterion2:
                connection_candidate.append(
                    [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
    connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
    return connection_candidate

def compute_connections(all_peaks, oriImg, paf):
    '''
    Computes connections
    '''
    connection_all = []
    special_k = []
    for k in range(len(MAP_IDX)):
        score_mid = paf[:, :, [x - 19 for x in MAP_IDX[k]]]
        candA = all_peaks[LIMB_SEQ[k][0] - 1]
        candB = all_peaks[LIMB_SEQ[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        if (nA != 0 and nB != 0):
            connection_candidate = search_connection_candidate(candA, candB, nA, nB, oriImg, score_mid)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    return connection_all, special_k


def compute_subset(all_peaks, connection_all, special_k):
    '''
    Computes subset
    '''
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    for k in range(len(MAP_IDX)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(LIMB_SEQ[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    return candidate, subset


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
        imageToTest_padded, pad = padRightDownCorner(frame)
        preprocessed_frame = preprocessing(imageToTest_padded, net.get_input_shape()[2], net.get_input_shape()[3])
        #print("Perform inference on the frame")
        net.async_inference(preprocessed_frame)

        heatmap = np.zeros((frame.shape[0], frame.shape[1], 19))
        paf = np.zeros((frame.shape[0], frame.shape[1], 38))
        if net.wait() == 0:
            # Get the output of inference
            output_blobs = net.extract_output()
            heatmap, paf = extract_outputs(imageToTest_padded, frame, output_blobs, pad)

        all_peaks = compute_peaks(heatmap)

        connection_all, special_k = compute_connections(all_peaks, frame, paf)

        candidate, subset = compute_subset(all_peaks, connection_all, special_k)

        cmap = matplotlib.cm.get_cmap('hsv')
        for i in range(18):
            rgba = np.array(cmap(1 - i/18. - 1./36))
            rgba[0:3] *= 255
            for j in range(len(all_peaks[i])):
                cv2.circle(frame, all_peaks[i][j][0:2], 4, COLORS[i], thickness=-1)

        # Adding sticks (lines) between the dots
        frame = adding_sticks_to_frame(frame, subset, candidate)

        # Write a frame here for debug purpose
        #cv2.imwrite("kungfu-frame" + str(frame_count) + ".png", frame)
     
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


def extract_outputs(imageToTest_padded, oriImg, output_blobs, pad):
    # extract outputs, resize, and remove padding
    # print("output_blobs.keys()=", output_blobs.keys())
    heatmap = np.transpose(np.squeeze(output_blobs['Mconv7_stage2_L2'].data),
                           (1, 2, 0))  # output Mconv7_stage2_L2 is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=STRIDE, fy=STRIDE, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    paf = np.transpose(np.squeeze(output_blobs['Mconv7_stage2_L1'].data), (1, 2, 0))  # output Mconv7_stage2_L1 is PAFs
    paf = cv2.resize(paf, (0, 0), fx=STRIDE, fy=STRIDE, interpolation=cv2.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    return heatmap, paf


def main():
    print("Starting")
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
