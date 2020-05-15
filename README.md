# human-pose-test

This a my personnal sandbox for test with various machine learning models in Python.
It started with tests on human pose detection (hence the name).

## Prerequisites

To use these scripts, you need Intel's OpenVino toolkit down installed on your machine and the OpenCV as well.
You can read more about both these.

- [OpenCV](https://opencv.org) - The simple install should look like ```pip install opencv-python```. 
- [OpenVino toolKit](https://software.intel.com/en-us/openvino-toolkit) - See website for installation depending of your configuration.

## Human pose

The scripts [main.py](main.py) and [mainpi.py](mainpi.py) are my (windows) desktop and raspberry pi tests on human pose detection.

## Emotion recognition

The scripts [emotion.py](emotion.py) and [emotionpi.py](emotionpi.py) are my (windows) desktop and raspberry pi tests on Emotion recognition.

## BERT

The script [bert.py](bert.py) is my test playing with similar questions recognition with BERT.

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture. It was originally published by Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova: "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)", 2018.

### Model used

I based my tests on the "multilingual_L-12_H-768_A-12" model that I found on the [Google Research Github page](https://github.com/google-research/bert/blob/master/multilingual.md).


## Cat detection

The script [cat.py](cat.py) is my test base of what became the [AceVINOtura](https://github.com/frankhn/AceVINOtura) project with help from some other Udacity students (see [acknowledgements](https://github.com/frankhn/AceVINOtura#acknowledgements)).

A lot of stuff is hardcoded in this script :
- the paths to the source video file used and the output file generated
- the path to the (windows) openvino CPU extension
- the path to the model (`frozen_inference_graph.xml`)
- the coordinates of the "forbidden zone" describing the zone where the cat should not go
- the confidence threshold for the model detections

### Model used

The model used (and **not** included in this repository) is a OpenVino IR converted from the [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) of the Tensorflow [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Conversion was made with the following command :
```
python mo_tf.py --input_model ssdlite_mobilenet_v2_coco\frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions\front\tf\ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco.config --data_type FP16
```

