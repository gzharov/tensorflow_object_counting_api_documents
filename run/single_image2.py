from utils2 import backbone
import image_detection

input_img = "UPD_72.jpg"


detection_graph, category_index = backbone.set_model('my_classifier', 'my_labelmap.pbtxt')

image_detection.detection_for_single_image(input_img, detection_graph, category_index, min_score_thresh=0.0)