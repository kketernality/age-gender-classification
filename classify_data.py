import os, sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# import better_exceptions
import caffe

DEVICE_ID = 0

class BasePredictor(object):
    def __init__(self):
        # load the mean ImageNet image
        mu = np.load('models/ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)
 
        # create transformer for input
        transformer = caffe.io.Transformer({'data': (1, 3, 224, 224)})
         
        transformer.set_transpose('data', (2,0,1))  
        transformer.set_mean('data', mu)            
        transformer.set_raw_scale('data', 255)      
        transformer.set_channel_swap('data', (2,1,0))

        self.transformer = transformer

class AgePredictor(BasePredictor):
    def __init__(self):
	if not os.path.isfile('models/age.caffemodel'):
            raise ValueError, 'Not found age.caffemodel'

        super(AgePredictor, self).__init__()
                   
        caffe.set_device(DEVICE_ID)
        caffe.set_mode_gpu()

        model_def = 'models/age.prototxt'
        model_weights = 'models/age.caffemodel'

        net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.net = net
                                                            
    def predict(self, image):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        output = self.net.forward()
        return output['prob'][0].argmax()
     
    def predict_fc8(self, image):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        self.net.forward()
        output_fc8 = self.net.blobs['fc8-101'].data
        return np.array(output_fc8, dtype=float) 
            
class GenderPredictor(BasePredictor):
    def __init__(self):
	if not os.path.isfile('models/gender.caffemodel'):
            raise ValueError, 'Not found gender.caffemodel' 

        super(GenderPredictor, self).__init__()

        caffe.set_device(DEVICE_ID)
        caffe.set_mode_gpu()

        model_def = 'models/gender.prototxt'
        model_weights = 'models/gender.caffemodel'

        net = caffe.Net(model_def, model_weights, caffe.TEST)                                                   
        self.net = net
                                                            
    def predict(self, image):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        output = self.net.forward()
        return output['prob'][0].argmax()
        
    def predict_fc8(self, image):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        self.net.forward()
        output_fc8 = self.net.blobs['fc8-2'].data
        return np.array(output_fc8, dtype=float) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    data_path = 'face'
    feat_path = 'feat'

    class_names = ['child_male', 'young_male', 'adult_male', 'elder_male',
                   'child_female', 'young_female', 'adult_female', 'elder_female']
    
    has_record = os.path.isfile(os.path.join(feat_path, 'feats.npy')) and \
                 os.path.isfile(os.path.join(feat_path, 'labels.npy'))
    
    age_predictor = AgePredictor()
    gender_predictor = GenderPredictor()	
    
    if has_record:
        print '[log] find previous record of features. Skip extraction'
    else:
        print '[log] start to extract features using CNNs'       
        print '[log] class nmaes include', class_names

        class_ages = dict([(class_name, []) for class_name in class_names])
        class_genders = dict([(class_name, []) for class_name in class_names])
        labels = []
        feats = []

        for class_name in class_names:
            print '[log] working on class {}'.format(class_name) 

            class_dir = os.path.join(data_path, class_name)
            for i, image_name in enumerate(os.listdir(class_dir)):
                image_path = os.path.join(class_dir, image_name)
                image = caffe.io.load_image(image_path)
                
                age_feat = age_predictor.predict_fc8(image).flatten()
                gender_feat = gender_predictor.predict(image).flatten()
                feat = np.concatenate([age_feat, gender_feat]).flatten()            

                labels.append(class_names.index(class_name))
                feats.append(feat)            
                if (i+1) % 100 == 0:  
                    print '[log] {} ith image completed'.format(i+1)

            print '[log] class {} completed'.format(class_name)

        print '[log] save extracted features to files'

        np.save(os.path.join(feat_path, 'labels.npy'), labels)
        np.save(os.path.join(feat_path, 'feats.npy'), feats)

    print '[log] load features and labels from files'

    labels = np.load(os.path.join(feat_path, 'labels.npy'))
    feats = np.load(os.path.join(feat_path, 'feats.npy'))

    print '[log] start fitting SVM to training features'

    # shuffle training data
    p = np.random.permutation(len(labels))
    feats = feats[p]
    labels = labels[p]

    clf = SVC(kernel='rbf', C=2**13, gamma=2**-10)

    print '[log] do 3-folds cross validation'
    print '[log] cross validation score:', cross_val_score(clf, feats, labels, n_jobs=-1)

    if not args.demo:
        exit()

    print '[log] demo time!'
    print '[log] fit the SVM with entire training set'

    clf.fit(feats, labels)

    print '[log] start demo. Predict label using SVM'

    demo_path = 'demo-face'

    result = np.zeros((640, 2), dtype=np.int)
    for i in range(640):
        result[i, 0] = i

    for i in range(640):
        image_path = os.path.join(demo_path, str(i) + '.jpg')

        if not os.path.isfile(image_path):
            
            label = np.random.randint(8)
            result[i, 1] = label
            print 'No image {} label {}'.format(i, label)

        else:
            image = caffe.io.load_image(image_path)
                
            age_feat = age_predictor.predict_fc8(image).flatten()
            gender_feat = gender_predictor.predict(image).flatten()
            feat = np.concatenate([age_feat, gender_feat]).flatten()            

            label = clf.predict(feat)

            result[i, 1] = label
            
            print 'image {} label {}'.format(i, label) 

    np.savetxt('result.csv', result, fmt='%d', delimiter=', ')

