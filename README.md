## :movie_camera:Objectdetection-MobileNetV3:camera_flash:
### Real time object detection in python using OpenCV, Mobile Net V3.

OpenCV has Deep Neural Network modules which is purely CPU based and no GPU is required.

There are many popular pretrained models with different kind of DNN architecture like YOLO, SDD, MobileNet, Faster-RNN, etc. 

##
I am using **MobileNetV3** trained on [COCO](https://arxiv.org/pdf/1405.0312.pdf) Dataset as its extremely small and lightweight (57mb). We can download the model from [TensorFlow Object Detection API wiki](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API). It includes datamodel (frozen_inference_graph.pb) and configuration file (pipeline.config) which we can use to generate the text graph (.pbtxt)

OpenCV needs an extra configuration file to import object detection models from TensorFlow. It's based on a text version of the same serialized graph in protocol buffers format (protobuf).
##
So to use the `cv2.dnn_Detectionmodel` we need weights(.pb) and config(.pbtxt) files. The model downloaded has the weights file. We can generate the config file from TensorFlow graph generation [tf_text_graph_ssd.py](https://github.com/opencv/opencv/blob/master/samples/dnn/tf_text_graph_ssd.py), (also requires [tf_text_graph_common.py](https://github.com/opencv/opencv/blob/master/samples/dnn/tf_text_graph_common.py)) by passing the address of input and config file: 

`python tf_text_graph_ssd.py --input frozen_inference_graph.pb --config pipeline.config --output ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`

Then I generated **coco_reformatted.names** file with the reference of [mscoco_label_map.pbtxt](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt). It will be passed as a list in our code and will be used to Label the objects.

##
![image](https://user-images.githubusercontent.com/54273763/123225531-71b72680-d4f0-11eb-8d86-c20c98a09022.png)
