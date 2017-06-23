# coding: utf-8
import cv2
import os
import time
import itertools
import os, sys
import argparse
import numpy as np

sys.path.append('mxnet_mtcnn_face_detection')
import mxnet as mx
from mtcnn_detector import MtcnnDetector

model_path = 'mxnet_mtcnn_face_detection/model'

size = 224
border = 0.4 

def compute_area(box):
    ''' Compute area of a bounding box '''
    return abs(box[0]-box[2])*abs(box[1]-box[3])

def compute_dist(box, image):
    ''' Compute the distance from center of bounding box to center of image '''
    h, w, _ = image.shape
    ch, cw = h//2, w//2
    bh, bw = (box[0]+box[2])//2, (box[1]+box[3])//2
    return np.sqrt((ch-bh)**2 + (cw-bw)**2)

detector = MtcnnDetector(model_folder=model_path, 
                         ctx=mx.cpu(0), 
                         num_worker=16, 
                         accurate_landmark=False)

parser = argparse.ArgumentParser()
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()

if not args.demo:
    data_path = 'data'
    face_path = 'face'

    # Create the same directory structure as the data path in face path
    print '[log] prepare destination face path {}'.format(face_path)

    if not os.path.exists(face_path):
        print '[log] create face directory'

        os.mkdir(face_path)
        for class_name in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, class_name)):
                os.mkdir(os.path.join(face_path, class_name))
    else:
        print '[log] clean data inside face directory'
        
        for class_name in os.listdir(data_path):
            class_face_path = os.path.join(face_path, class_name)
            if os.path.isdir(os.path.join(data_path, class_name)):
                if not os.path.exists(class_face_path):
                    os.mkdir(class_face_path)
                else:
                    for image_name in os.listdir(class_face_path):
                        os.remove(os.path.join(class_face_path, image_name))

    # Start to crop and align face using the code from MTCNN implementation
    print '[log] start to process training data'
    print '[log] crop and align face from data path {} to face path {}'.format(data_path, face_path)

    all_cor, all_mis = 0, 0 # Statistic for face detection rate

    for class_name in os.listdir(data_path):
        if not os.path.isdir(os.path.join(data_path, class_name)):
            continue

        print '[log] start to process data in {}'.format(class_name)
        class_path = os.path.join(data_path, class_name)

        cor, mis = 0, 0
        
        for i, image_name in enumerate(os.listdir(class_path)):
            if (i+1) % 100 == 0:
                print '[log] processing {} th images'.format(i+1) 

            image = cv2.imread(os.path.join(class_path, image_name))
            
            results = detector.detect_face(image) # This use mtcnn to detect face

            if results is not None:
                boxes, points = results # Bouding boxes and their landmark points
                
                if len(boxes) > 1:
                    print '[log] more than one face found in {}'.format(' '.join([class_name, image_name]))

                    boxes_dist = [compute_dist(box, image) for box in boxes]
                    boxes_area = [compute_area(box) for box in boxes]

                    # Select the bounding box with the largest area. One can apply other policy as well
                    boxes = [boxes[np.argmax(boxes_area)]]
                    points = [points[np.argmax(boxes_area)]]

                chips = detector.extract_image_chips(image, points, size, border)
                assert len(chips) == 1

                write_path = os.path.join(face_path, class_name, image_name)
                cv2.imwrite(write_path, chips[0])

                cor += 1  # Detected
            else:
                print '[log] no face found in {}'.format(' '.join([class_name, image_name]))
                    
                mis += 1  # Not detected

        all_cor, all_mis = all_cor+cor, all_mis+mis

        print '[log] successully found {:5.2f} % face inside {}'.format(100*cor/float(cor+mis), class_name)

    print '[log] successfully found {:5.2f} % face in total'.format(100*all_cor/float(all_cor+all_mis))

else:
    data_path = 'demo-data'
    face_path = 'demo-face'

    print '[log] start to process demo data'
    print '[log] crop and align face from data path {} to face path {}'.format(data_path, face_path)

    cor, mis = 0, 0

    for i in range(640):
        
        image = cv2.imread(os.path.join(data_path, str(i) + '.jpg'))
        
        results = detector.detect_face(image)
        
        if results is not None:
            
            total_boxes = results[0]
            points = results[1]

            if len(total_boxes) > 1:

                print '[log] more than one face found in {}'.format(str(i) + '.jpg')

                boxes_dist = [compute_dist(box, image) for box in total_boxes]
                boxes_area = [compute_area(box) for box in total_boxes]
            
                total_boxes = [total_boxes[np.argmax(boxes_area)]]
                points = [points[np.argmax(boxes_area)]]


            chips = detector.extract_image_chips(image, points, size, border)
            assert len(chips) == 1

            write_path = os.path.join(face_path, str(i) + '.jpg')

            cv2.imwrite(write_path, chips[0])

            cor += 1
        else:

            print '[log] no face found in {}'.format(str(i) + '.jpg')
            
            mis += 1

    print '[log] successfully found {} or {:5.2f} % face'.format(cor, 100.0*cor/float(cor + mis))

