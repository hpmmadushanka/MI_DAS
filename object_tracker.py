from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools
import shutil
from data import cfg, set_cfg, set_dataset
from yolact import Yolact



import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json

from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
import sys
import subprocess



import os
import time, random
import numpy as np
from numpy import *
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing


from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

global total
def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        #print(fps)
        #print(width)
        #print(height)
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    
    fps = 0.0
    count = 0 
    tot = -1
    
	

    coord = {}
    pred = {}
    framelists = []
    frame_count = 0
    while True:
        frame_area = width*height

        #print("count" + str(count))
        tot = tot + 1
        total =tot
        
        _, img = vid.read()
        #print("length of framelists " +str(len(framelists)))
        #if(15<tot<55):
            #cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(250,0,0))
        #if(tot<-6):
            #cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(0,0,0))
       
        
        temp_img = img
        #count = 0
        temp_framelists = []
        if(len(framelists)==5):
            framelists.remove(framelists[0])
            for i in range(0,5):
                img_v = cv2.imread("data/video/raw1/frame%d.jpg" % i)
                temp_framelists.append(img_v)
            temp_framelists.remove(temp_framelists[0])
            temp_framelists.append(temp_img)

            for i in range(0,len(temp_framelists)):
                cv2.imwrite("data/video/raw1/frame%d.jpg" % i, temp_framelists[i]) 
       
            
        else:

            if(count <5):
                if(count ==0):
                    cv2.imwrite("data/video/raw1/frame0.jpg", img ) 
                    #print(1)

                else:
                    cv2.imwrite("data/video/raw1/frame%d.jpg" % count, temp_img) 
                    #print(2)

            for i in range(0,count+1):
                img_v = cv2.imread("data/video/raw1/frame%d.jpg" % i)
                temp_framelists.append(img_v)
        framelists.append(temp_img)
        

        #if(len(framelists) >0):
            #cv2.imwrite("data/video/raw/frame%d.jpg" % tot, framelists[-1]) 
        
        
       
        
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
        		
		
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        	 
		 
        #if(90<tot<115):
            #cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(255,0,0))
		
		
		
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        #print(detections)
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


        
        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]     
        #print(detections)   

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        #####
        
        ls = [] #to get the track ids of each frame

        

        
        a = framelists[-1]

        
             
        
        #cv2.imwrite("data/video/raw1/frame%d.jpg" % tot, temp_framelists[-1]) 
        
        #print("scores " + str(scores))
        #print("len scores " + str(len(scores)))
        #print("len detections " + str(len(tracker.tracks)))
        index =-1
        i =-1
        for track in tracker.tracks:
            i=i+1
            #print("track " + str(tracker.tracks))
            
            
            x_min = []
            x_max = []
            y_max = []
            
            temp_box = track.to_tlbr()
            ls.append(track.track_id)#add track ids
            if track.track_id not in coord:
                coord[track.track_id] = []
            else:
                
                leng = len(coord[track.track_id])
                if(leng == 5):
                    coord[track.track_id].remove(coord[track.track_id][0])
                coord[track.track_id].append(list(temp_box)) 
            #print("*****" + str(coord[track.track_id]))
            
            for i in range(0,len(coord[track.track_id])):
                if(coord[track.track_id][i][0]>0 and coord[track.track_id][i][2]<720 and coord[track.track_id][i][3]<720):
                    x_min.append(coord[track.track_id][i][0])
                    x_max.append(coord[track.track_id][i][2])
                    y_max.append(coord[track.track_id][i][3])
			
            if(len(coord[track.track_id])>0):			
                vehicle_area  =  abs((coord[track.track_id][0][2]  -  coord[track.track_id][0][0]) * (coord[track.track_id][0][3] - coord[track.track_id][0][1]))
                percentage = vehicle_area/frame_area
            else:
                percentage = 0

            #print(x_min)
            #print(x_max)
            #print(y_max)
            base = []
            #print(coord[track.track_id])
            #cv2.imwrite("data/video/raw1/image%d.jpg" % tot, a) 
            if(len(x_min)>=3 and tot>4):
                for i in range(0,len(x_min)):
                    base.append(i+1)
            
            
                model1 = polyfit(base,x_min,1)
                predict_x_min = poly1d(model1)

                model2 = polyfit(base,x_max,1)
                predict_x_max = poly1d(model2)

                model3 = polyfit(base,y_max,1)
                predict_y_max = poly1d(model3)
                
                tempo_framelists = []
                
                #if(90<count<125):
                    #cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(0,0,0))
                
                if((50<predict_x_min(60)<650 and 50<predict_x_max(60)<650)and 400<predict_y_max(60) and 0.005<percentage<0.5):
                    cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(0,0,0))
                    
                
                else:
                    #if((50<predict_x_min(60)<650 or 50<predict_x_max(60)<650) and 500<predict_y_max(60)<800 ):
                    if((50<predict_x_min(60)<650 or 50<predict_x_max(60)<650) and 400<predict_y_max(60) and 0.005<percentage<0.5):
                        cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(255,0,0))
                        
                    
                        for i in range(0,len(framelists)):
                            image_vehicle = cv2.imread("data/video/raw1/frame%d.jpg" % i) 
                            tempo_framelists.append(image_vehicle)
                        
                        pos =[]

                        #print("length of framelists : "+ str(len(tempo_framelists)))
                        #print("length of coord : "+ str(len(coord[track.track_id])))
                                            

                        try:
                            dimlist = []
                                                    
                            for j in range(0,len(coord[track.track_id])):
                                
                                #print("len" + str(len(coord[track.track_id])))                        
                                vehicle = coord[track.track_id][-(j+1)]
                                #print(framelists)
                                
                                crop_img = tempo_framelists[-(j+1)][int(vehicle[1])-10:int(vehicle[3])+10, int(vehicle[0])-10:int(vehicle[2])+10]
                                inp_img =  tempo_framelists[-(j+1)]
                                crop_height = crop_img.shape[0]
                                crop_width = crop_img.shape[1]
                                

                                #print("crops" + str(crop_height) + "," + str(crop_width))
                                #print(int(vehicle[3])- int(vehicle[1]), int(vehicle[2])- int(vehicle[0]))


                                input_dir = os.path.join("data/video/cropped/turn%d" %tot)
                                output_dir = os.path.join("data/video/segment/turn%d" %tot)
                                if not os.path.exists(input_dir):
                                    os.mkdir(input_dir)    
                                    os.mkdir(output_dir)                    

                                #cv2.imwrite("data/video/cropped/frame%d.jpg" % j,crop_img )
                            

                                #input_image = "data/video/cropped/frame%d.jpg" % j
                                #output_image ="data/video/segment/frame%d.jpg" % j
                                input_folder = "data/video/cropped/turn%d" %tot
                                output_folder = "data/video/segment/turn%d" %tot
                               
                                input_image = input_folder  +  "/frame%d.jpg" % j
                                output_image = output_folder + "/frame%d.jpg" % j
                                dim = []
                                #cv2.imwrite(input_image ,crop_img )
                                cv2.imwrite(input_image ,inp_img) 
                                xmin = int(vehicle[0])-10
                                dim.append(xmin)
                                xmax = int(vehicle[2])+10
                                dim.append(xmax)
                                ymin = int(vehicle[1])-10
                                dim.append(ymin)
                                ymax = int(vehicle[3])+10
                                dim.append(ymax)
                                                            
                                #ymax = str(int(vehicle[3])+10)
                                #dim = xmin + ","+ xmax + "," + ymin + "," +ymax
                                dimlist.append(dim)
                                
                                                            
                                #if condition for restricted area for segmentation
                            
                               
                            if((40<predict_x_min(60)<670 and 40<predict_x_max(60)<670) or frame_count>0 or tot<-1):
                                cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(0,0,0))
                                frame_count = frame_count-1
                            elif(count<-2):
                               
                                                            
                                dimlist = str(dimlist)
                                dimlist = dimlist.replace(" ","")
                                os.system("python eval1.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=" + input_folder+ ":" +output_folder  +" --dimension=" +dimlist)
                                                            
                                file1 = open('positions.txt', 'r')
                                Lines = file1.readlines()
                                pos = []
                                for line in Lines:
                                    points = line.split(",")
                                    pos.append(points)
                                pos.reverse()
                                final_pred = []	
                                if(len(pos)>3):
                                    if(len(pos[-1])!=8):
                                        for i in range(0,len(pos[0])):
                                            evalu_x = []
                                            evalu_y = []
                                            for k in range(0,len(pos)):
                                                evalu_y.append(int(pos[k][i]))
                                                evalu_x.append(k)
                                            model = polyfit(evalu_x,evalu_y,2)
                                            predict = poly1d(model)
                                            final_pred.append(predict(60))
                                    if((50<final_pred[2]<630 or 50<final_pred[4]<630) and (400<final_pred[3]<800 or 500<final_pred[5]<800)):
                                        cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=8,color=(0,0,0))
                                        frame_count = 15
							

                        except Exception as e:
                            print("except")
                            print(e)
                            continue
                    
                
            #if(track.track_id ==13):
                #print("temp  " + str(temp_box ))
            #if not track.is_confirmed() or track.time_since_update > 1:
                #continue 
			
            bbox = track.to_tlbr()
            
            #if(track.track_id ==13):
                #print("special  " + str(bbox))
            #print("box is " + str(bbox))
            #print(track.track_id , bbox)	
            #print(track.track_id)	
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
			
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
		
		
        count = count+1
        #print("track ids : " +str(ls))  
        #print("dict :   " + str(coord))
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        #for det in detections:
        #    bbox = det.to_tlbr() 
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        # print fps on screen 
        #fps  = ( fps + (1./(time.time()-t1)) ) / 2
        
						  
        #cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        cv2.imshow('output', img)
        #if tot == 6 :
            
            #cv2.imwrite("data/video/images/000.jpg",framelists[0])
            #cv2.imwrite("data/video/images/001.jpg",framelists[1])
            #cv2.imwrite("data/video/images/002.jpg",framelists[2])
            #cv2.imwrite("data/video/images/003.jpg",framelists[3])
            #cv2.imwrite("data/video/images/004.jpg",framelists[4])
        
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')
            
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.ouput:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    
    try:
        app.run(main)
    except SystemExit:
        pass
