import tensorflow as tf
import csv
import cv2
import numpy as np



import collections
import functools
import matplotlib.pyplot as plt
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import numpy
import os




##---------------------------------------------- draw


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='lime',
                                     thickness=4,
                                     display_str_list=(),
                                     ):
  """Adds a bounding box to an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box in normalized coordinates (same below).
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list)
  np.copyto(image, np.array(image_pil))
  
  
  
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='lime',
                               thickness=4,
                               display_str_list=()
                               ):
  """Adds a bounding box to an image.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  #print('im_width, im_height = ',im_width, im_height)
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    #print('lfrb = ',left, right, top, bottom)
  
  try:
    font = ImageFont.truetype('arial.ttf', 16)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_list[0] = display_str_list[0]
  #csv_line = str (predicted_direction) # csv line created
  
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    #print('text_width, text_height, margin = ',text_width, text_height, margin)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
    
    
    
##---------------------------------------------------------------------------------------------
    
    
def get_greed_rectangles(img_width, img_height, rectangle_shapes, shift_forward, shift_down):
    
    greed_rectangles = []
    right_bottom_angle = [rectangle_shapes[0], rectangle_shapes[1]]
    
    
    while (right_bottom_angle[1]<=img_height):
        
        greed_rectangles.append([right_bottom_angle[0]-rectangle_shapes[0], right_bottom_angle[0], right_bottom_angle[1] - rectangle_shapes[1] , right_bottom_angle[1]])
        
        right_bottom_angle[0]+=shift_forward
        
        if right_bottom_angle[0] > img_width:
            right_bottom_angle[0] = rectangle_shapes[0]
            right_bottom_angle[1] = right_bottom_angle[1]+shift_down
            
    return greed_rectangles
    

##указываются части от ширины и высоты листа как соотношение
def get_greed_rectangles_universal(img_width, img_height, rectangle_shapes_part, shift_forward_part, shift_down_part):
    
    rectangle_shapes = [0,0]
    rectangle_shapes[0] = round(img_width/rectangle_shapes_part[0])
    rectangle_shapes[1] = round(img_height/rectangle_shapes_part[1])
    shift_forward = round(img_width/shift_forward_part)
    shift_down = round(img_height/shift_down_part)
    
    
    if img_width<img_height:
        t = rectangle_shapes[0]
        rectangle_shapes[0] = rectangle_shapes[1]
        rectangle_shapes[1] = t
        
        t2 = shift_forward
        shift_forward = shift_down
        shift_down = t2
        
    
    
    greed_rectangles = []
    right_bottom_angle = [rectangle_shapes[0], rectangle_shapes[1]]
    
    
    while (right_bottom_angle[1]<=img_height):
        
        greed_rectangles.append([right_bottom_angle[0]-rectangle_shapes[0], right_bottom_angle[0], right_bottom_angle[1] - rectangle_shapes[1] , right_bottom_angle[1]])
        
        right_bottom_angle[0]+=shift_forward
        
        if right_bottom_angle[0] > img_width:
            right_bottom_angle[0] = rectangle_shapes[0]
            right_bottom_angle[1] = right_bottom_angle[1]+shift_down
            
    return greed_rectangles
    
    
##----------------------------------------
    

def show_detection_for_single_image(input_img, detection_graph, category_index, min_score_thresh=0.5):

        names_classes = ['TORG12','UPD','UPD2','factura']
        output_path = 'results/'
        name = input_img.replace('/',' ').split()[-1]
        
        
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')            

       

            input_frame = cv2.imread(input_img)
            image_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            #print(input_frame.shape)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_rgb, axis=0)
            #print(image_np_expanded)

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
                
                
            
            #select objects
            
            img_boxes = np.reshape(boxes, (len(boxes[0]), 4))
            img_scores = np.reshape(scores,(len(boxes[0])))
            img_classes = np.reshape(classes,(len(boxes[0])))
            #print(img_boxes, img_scores, img_classes)
            
            selected_boxes=[]
            selected_scores=[]
            selected_classes=[]
            
            for i in range(len(img_boxes)):
                #print(float(img_scores[i]))
                if img_scores[i] > min_score_thresh:
                    selected_boxes.append(img_boxes[i].tolist())
                    selected_scores.append(img_scores[i])
                    selected_classes.append(img_classes[i])
                
            #print(selected_boxes, selected_scores, selected_classes)
            
            
            for i in range(len(selected_boxes)):
                ymin,xmin,ymax,xmax = selected_boxes[i][0], selected_boxes[i][1], selected_boxes[i][2], selected_boxes[i][3]
                #display_str_list = ["ID: "+str(track_id[i])]
                new_score = selected_scores[i]*100
                dsl = [names_classes[int(selected_classes[i])-1] + ": " + str(new_score) + "%"]
                draw_bounding_box_on_image_array(input_frame,
                                    ymin,
                                    xmin,
                                    ymax,
                                    xmax,
                                    display_str_list = dsl)
                                    
                                    
            cv2.imwrite(output_path + name,input_frame)
            print("processed: "+name)
            cv2.waitKey(0)
            
##-----------------------------------------
            
            
def show_single_img_with_greed(input_img, detection_graph, category_index, greed_params, min_score_thresh=0.5):
     
    names_classes = ['TORG12','UPD','UPD2', 'factura']
    output_path = 'results2/'
    #name = input_img.replace('/',' ').split()[-1]
    
    input_frame = cv2.imread(input_img)
    width = input_frame.shape[1]
    height =input_frame.shape[0]
    
    with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            #greed
            #greed = get_greed_rectangles(width, height, [1000,1000] , 500 , 500)
            greed = get_greed_rectangles(width, height, [greed_params[0],greed_params[1]] , greed_params[2] , greed_params[3])
            #print(greed)
            
            for j in range(len(greed)):
                x1 = greed[j][0]
                x2 = greed[j][1]
                y1 = greed[j][2]
                y2 = greed[j][3]
                crop_img = input_frame[y1:y2, x1:x2]
                #print(crop_img)
        
                image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_rgb, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        
                #select objects
                img_boxes = np.reshape(boxes, (len(boxes[0]), 4))
                img_scores = np.reshape(scores,(len(boxes[0])))
                img_classes = np.reshape(classes,(len(boxes[0])))
                #print(img_boxes, img_scores, img_classes)
            
                selected_boxes=[]
                selected_scores=[]
                selected_classes=[]
                
                for i in range(len(img_boxes)):
                    #print(float(img_scores[i]))
                    if img_scores[i] > min_score_thresh:
                        selected_boxes.append(img_boxes[i].tolist())
                        selected_scores.append(img_scores[i])
                        selected_classes.append(img_classes[i])
                
                #print(selected_boxes, selected_scores, selected_classes)
            
            
                for i in range(len(selected_boxes)):
                    ymin,xmin,ymax,xmax = selected_boxes[i][0], selected_boxes[i][1], selected_boxes[i][2], selected_boxes[i][3]
                    #display_str_list = ["ID: "+str(track_id[i])]
                    new_score = round(selected_scores[i]*100)
                    dsl = [names_classes[int(selected_classes[i])-1] + ": " + str(new_score) + "%"]
                    draw_bounding_box_on_image_array(crop_img,
                                    ymin,
                                    xmin,
                                    ymax,
                                    xmax,
                                    display_str_list = dsl)
                                    
                                    
                cv2.imwrite(output_path + str(j+1) + '.jpg', crop_img)
                print("processed: " + str(j+1))
    
    
            
            
##-------------------------------------


def show_single_img_with_greed_universal(input_img, detection_graph, category_index, greed_params, min_score_thresh=0.5):
     
    names_classes = ['TORG12','UPD','UPD2', 'factura']
    output_path = 'results2/'
    #name = input_img.replace('/',' ').split()[-1]
    
    input_frame = cv2.imread(input_img)
    width = input_frame.shape[1]
    height =input_frame.shape[0]
    
    with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            #greed
            #greed = get_greed_rectangles(width, height, [1000,1000] , 500 , 500)
            greed = get_greed_rectangles_universal(width, height, [greed_params[0],greed_params[1]] , greed_params[2] , greed_params[3])
            #print(greed)
            
            for j in range(len(greed)):
                x1 = greed[j][0]
                x2 = greed[j][1]
                y1 = greed[j][2]
                y2 = greed[j][3]
                crop_img = input_frame[y1:y2, x1:x2]
                #print(crop_img)
        
                image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_rgb, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        
                #select objects
                img_boxes = np.reshape(boxes, (len(boxes[0]), 4))
                img_scores = np.reshape(scores,(len(boxes[0])))
                img_classes = np.reshape(classes,(len(boxes[0])))
                #print(img_boxes, img_scores, img_classes)
            
                selected_boxes=[]
                selected_scores=[]
                selected_classes=[]
                
                for i in range(len(img_boxes)):
                    #print(float(img_scores[i]))
                    if img_scores[i] > min_score_thresh:
                        selected_boxes.append(img_boxes[i].tolist())
                        selected_scores.append(img_scores[i])
                        selected_classes.append(img_classes[i])
                
                #print(selected_boxes, selected_scores, selected_classes)
            
            
                for i in range(len(selected_boxes)):
                    ymin,xmin,ymax,xmax = selected_boxes[i][0], selected_boxes[i][1], selected_boxes[i][2], selected_boxes[i][3]
                    #display_str_list = ["ID: "+str(track_id[i])]
                    new_score = round(selected_scores[i]*100)
                    dsl = [names_classes[int(selected_classes[i])-1] + ": " + str(new_score) + "%"]
                    draw_bounding_box_on_image_array(crop_img,
                                    ymin,
                                    xmin,
                                    ymax,
                                    xmax,
                                    display_str_list = dsl)
                                    
                                    
                cv2.imwrite(output_path + str(j+1) + '.jpg', crop_img)
                print("processed: " + str(j+1))
    
    
            
            
##-------------------------------------


def classification_single_img_with_greed(input_img, detection_graph, category_index, greed_params, min_score_thresh=0.5):
    names_classes = ['TORG12','UPD','UPD2', 'factura']
    output_path = 'results2/'
    #name = input_img.replace('/',' ').split()[-1]
    
    input_frame = cv2.imread(input_img)
    width = input_frame.shape[1]
    height =input_frame.shape[0]
    
    with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            #greed
            #greed = get_greed_rectangles(width, height, [1000,1000] , 500 , 500)
            
            greed = get_greed_rectangles(width, height, [greed_params[0],greed_params[1]] , greed_params[2] , greed_params[3])
            
            
            counter = 0
            while(counter<len(greed)):
                x1 = greed[counter][0]
                x2 = greed[counter][1]
                y1 = greed[counter][2]
                y2 = greed[counter][3]
                crop_img = input_frame[y1:y2, x1:x2]
                #print(crop_img)
        
                image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_rgb, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        
                #select objects
                img_boxes = np.reshape(boxes, (len(boxes[0]), 4))
                img_scores = np.reshape(scores,(len(boxes[0])))
                img_classes = np.reshape(classes,(len(boxes[0])))
                #print(img_boxes, img_scores, img_classes)
            
                selected_boxes=[]
                selected_scores=[]
                selected_classes=[]
                
                for i in range(len(img_boxes)):
                    #print(float(img_scores[i]))
                    if img_scores[i] > min_score_thresh:
                        selected_boxes.append(img_boxes[i].tolist())
                        selected_scores.append(img_scores[i])
                        selected_classes.append(img_classes[i])
                
                if len(selected_boxes)>0:
                    print(names_classes[int(selected_classes[0])-1], selected_scores[0])
                    return names_classes[int(selected_classes[0])-1]
                    
                counter += 1
            
            print('other')    
            return 'other'
                                    


##--------------------------------------------


def classification_single_img_with_greed_universal(input_img, detection_graph, category_index, greed_params, min_score_thresh=0.9):
    names_classes = ['TORG12','UPD','UPD2', 'factura']
    output_path = 'results2/'
    #name = input_img.replace('/',' ').split()[-1]
    
    input_frame = cv2.imread(input_img)
    width = input_frame.shape[1]
    height =input_frame.shape[0]
    
    with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            #greed
            #greed = get_greed_rectangles(width, height, [1000,1000] , 500 , 500)
            
            greed = get_greed_rectangles_universal(width, height, [greed_params[0],greed_params[1]] , greed_params[2] , greed_params[3])
            
            
            counter = 0
            while(counter<len(greed)):
                x1 = greed[counter][0]
                x2 = greed[counter][1]
                y1 = greed[counter][2]
                y2 = greed[counter][3]
                crop_img = input_frame[y1:y2, x1:x2]
                #print(crop_img)
        
                image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_rgb, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        
                #select objects
                img_boxes = np.reshape(boxes, (len(boxes[0]), 4))
                img_scores = np.reshape(scores,(len(boxes[0])))
                img_classes = np.reshape(classes,(len(boxes[0])))
                #print(img_boxes, img_scores, img_classes)
            
                selected_boxes=[]
                selected_scores=[]
                selected_classes=[]
                
                for i in range(len(img_boxes)):
                    #print(float(img_scores[i]))
                    if img_scores[i] > min_score_thresh:
                        selected_boxes.append(img_boxes[i].tolist())
                        selected_scores.append(img_scores[i])
                        selected_classes.append(img_classes[i])
                
                if len(selected_boxes)>0:
                    print(names_classes[int(selected_classes[0])-1], selected_scores[0])
                    return names_classes[int(selected_classes[0])-1]
                    
                counter += 1
            
            print('other')    
            return 'other'
                                    


##--------------------------------------------


def classification_single_img_with_greed_long(input_img, detection_graph, category_index, greed_params, min_score_thresh=0.5):
    names_classes = ['TORG12','UPD','UPD2', 'factura']
    output_path = 'results2/'
    #name = input_img.replace('/',' ').split()[-1]
    
    input_frame = cv2.imread(input_img)
    width = input_frame.shape[1]
    height =input_frame.shape[0]
    
    with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            
            greed = get_greed_rectangles(width, height, [greed_params[0],greed_params[1]] , greed_params[2] , greed_params[3])
            
            max_score = -1
            point = ' '
            for j in range(len(greed)):
                x1 = greed[j][0]
                x2 = greed[j][1]
                y1 = greed[j][2]
                y2 = greed[j][3]
                crop_img = input_frame[y1:y2, x1:x2]
                #print(crop_img)
        
                image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_rgb, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        
                #select objects
                img_boxes = np.reshape(boxes, (len(boxes[0]), 4))
                img_scores = np.reshape(scores,(len(boxes[0])))
                img_classes = np.reshape(classes,(len(boxes[0])))
                #print(img_boxes, img_scores, img_classes)
            
                selected_boxes=[]
                selected_scores=[]
                selected_classes=[]
                
                for i in range(len(img_boxes)):
                    #print(float(img_scores[i]))
                    if img_scores[i] > min_score_thresh:
                        selected_boxes.append(img_boxes[i].tolist())
                        selected_scores.append(img_scores[i])
                        selected_classes.append(img_classes[i])
                
                if len(selected_boxes)>0:
                    if selected_scores[0]>max_score:
                        point = names_classes[int(selected_classes[0])-1]
                        max_score = selected_scores[0]
                        
                
                    
                
            
            if point == ' ':
                print('ohter')
                return 'other'
            else:
                print(point, max_score)
                return point
                
##-----------------------------------------------------



                
                

def classification_single_img_with_greed_long_universal(input_img, detection_graph, category_index, greed_params, min_score_thresh=0.5):
    names_classes = ['TORG12','UPD','UPD2', 'factura']
    output_path = 'results2/'
    #name = input_img.replace('/',' ').split()[-1]
    
    input_frame = cv2.imread(input_img)
    width = input_frame.shape[1]
    height =input_frame.shape[0]
    
    with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            
            greed = get_greed_rectangles_universal(width, height, [greed_params[0],greed_params[1]] , greed_params[2] , greed_params[3])
            
            max_score = -1
            point = ' '
            for j in range(len(greed)):
                x1 = greed[j][0]
                x2 = greed[j][1]
                y1 = greed[j][2]
                y2 = greed[j][3]
                crop_img = input_frame[y1:y2, x1:x2]
                #print(crop_img)
        
                image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_rgb, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        
                #select objects
                img_boxes = np.reshape(boxes, (len(boxes[0]), 4))
                img_scores = np.reshape(scores,(len(boxes[0])))
                img_classes = np.reshape(classes,(len(boxes[0])))
                #print(img_boxes, img_scores, img_classes)
            
                selected_boxes=[]
                selected_scores=[]
                selected_classes=[]
                
                for i in range(len(img_boxes)):
                    #print(float(img_scores[i]))
                    if img_scores[i] > min_score_thresh:
                        selected_boxes.append(img_boxes[i].tolist())
                        selected_scores.append(img_scores[i])
                        selected_classes.append(img_classes[i])
                
                if len(selected_boxes)>0:
                    if selected_scores[0]>max_score:
                        point = names_classes[int(selected_classes[0])-1]
                        max_score = selected_scores[0]
                        
                
                    
                
            
            if point == ' ':
                print('ohter')
                return 'other'
            else:
                print(point, max_score)
                return point



##---------------------------------------------------------------




def classification_long_with_location_universal(input_img, detection_graph, category_index, greed_params, location_dict = None, min_score_thresh=0.5):
    names_classes = ['TORG12','UPD','UPD2', 'factura']
    output_path = 'results2/'
    #name = input_img.replace('/',' ').split()[-1]
    
    input_frame = cv2.imread(input_img)
    width = input_frame.shape[1]
    height =input_frame.shape[0]
    
    flag = True
    if location_dict is None:
        flag = False
    
    with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            
            greed = get_greed_rectangles_universal(width, height, [greed_params[0],greed_params[1]] , greed_params[2] , greed_params[3])
            
            max_score = -1
            point = ' '
            index_location = -1
            
            for j in range(len(greed)):
                x1 = greed[j][0]
                x2 = greed[j][1]
                y1 = greed[j][2]
                y2 = greed[j][3]
                crop_img = input_frame[y1:y2, x1:x2]
                #print(crop_img)
        
                image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_rgb, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        
                #select objects
                img_boxes = np.reshape(boxes, (len(boxes[0]), 4))
                img_scores = np.reshape(scores,(len(boxes[0])))
                img_classes = np.reshape(classes,(len(boxes[0])))
                #print(img_boxes, img_scores, img_classes)
            
                selected_boxes=[]
                selected_scores=[]
                selected_classes=[]
                
                for i in range(len(img_boxes)):
                    #print(float(img_scores[i]))
                    if img_scores[i] > min_score_thresh:
                        selected_boxes.append(img_boxes[i].tolist())
                        selected_scores.append(img_scores[i])
                        selected_classes.append(img_classes[i])
                
                if len(selected_boxes)>0:
                    if selected_scores[0]>max_score:
                        
                        if flag:
                            point_t = names_classes[int(selected_classes[0])-1]
                            if ( j+1 in location_dict[point_t] ):
                                point = point_t
                                max_score = selected_scores[0]
                                index_location = j
                                #print(max_score)
                            
                        else:
                            point = names_classes[int(selected_classes[0])-1]
                            max_score = selected_scores[0]
                        
                
                    
                
            
            if point == ' ':
                print('ohter')
                return 'other'
            else:
                print(point, max_score, 'index: ',index_location + 1)
                return point

            
            
          
        
            








            
            
