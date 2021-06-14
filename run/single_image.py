from utils2 import backbone
from api import object_counting_api

#input_video = "test_videos/3.mp4"
#input_video = "test_videos/sub-1504619634606.mp4"
#input_video = "test_videos/2.1.mp4"
input_img = "UPD_72-crop.jpg"



#indicate the name of the directory where frozen_inference_graph.pb is located and the label map in data folder
#detection_graph, category_index = backbone.set_model('faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28', 'mscoco_label_map.pbtxt')
detection_graph, category_index = backbone.set_model('my_classifier', 'my_labelmap.pbtxt')

targeted_objects = "TORG12, UPD, UPD2" # (for counting targeted objects) change it with your targeted objects
#targeted_objects = "vehicle" # (for counting targeted objects) change it with your targeted objects

is_color_recognition_enabled = 0

object_counting_api.single_image_object_counting(input_img, detection_graph, category_index, is_color_recognition_enabled)