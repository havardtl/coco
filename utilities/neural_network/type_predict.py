import os
#import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import backend as K
from keras.models import load_model
import numpy as np

def get_classes(path = None):
    '''
    Get classes used in prediction
    
    Params
    path : str : path to text file with classes. If set to None, looks for it in default location
    
    Returns
    classes : list of str : the class description strings in order used in neural network
    '''
    
    if path is None: 
        train_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"AI_train_results")
    else: 
        train_data_folder = path 
    
    classes_path = os.path.join(train_data_folder,"classes.txt")
    with open(classes_path,'r') as f: 
        classes = f.readlines()

    for i in range(0,len(classes)):
        classes[i] = classes[i].strip()
        
    return classes

def get_model(path = None): 
    '''
    Get pre-trained neural network model used to predict types
    
    Params
    path : str : path to model file. If set to None, looks for it in default location
    
    Returns
    model : Keras model : The neural network model
    '''

    if path is None:  
        train_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"AI_train_results")
    else: 
        train_data_folder = path 

    model_path = os.path.join(train_data_folder,"last_model.h5")
    model = load_model(model_path)
    
    return model

def get_keras_data_format(): 
    return K.image_data_format()
    
def get_predictions(objects,keras_data_format = "channels_last"): 
    '''
    Predict the type of object from images of it 
    
    Params
    objects           : list of np.array : List of 120x120 grayscale images to predict
    keras_data_format : str              : keras data format used in training model. 
    
    Returns 
    predictions : list of str : Predictions 
    '''
    
    objects_correct_format = []
    for o in objects: 
        if keras_data_format == "channels_first": 
            o = o[np.newaxis,...]
        elif keras_data_format == "channels_last": 
            o = o[...,np.newaxis]
        objects_correct_format.append(o) 
        
    model = get_model()
    classes = get_classes()
    objects_correct_format = np.array(objects_correct_format)
    
    model_predictions = model.predict(objects_correct_format)
    
    predictions = []
    for p in model_predictions: 
        type = classes[np.argmax(p)]
        predictions.append(type)
    
    return predictions
    
