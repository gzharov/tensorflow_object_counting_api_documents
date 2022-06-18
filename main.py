from preprocessing import rotation
from utils2 import backbone
import image_detection

dir_path = 'test_complectation'

detection_graph, category_index = backbone.set_model('my_classifier', 'my_labelmap.pbtxt')

universal_greed_params = [3.5,2.48,7,4.95]

location_dict = {'UPD':[1], 'UPD2':[1], 'factura':[1,2], 'TORG12':[4,5,6,7,8,9,10,11,12,13,14]}

#rotation.rotation_dir(dir_path)

list_res = image_detection.complectation_dir(dir_path,  detection_graph, category_index, universal_greed_params, location_dict = location_dict, min_score_thresh=0.5)
#image_detection.write_file_csv(list_res)