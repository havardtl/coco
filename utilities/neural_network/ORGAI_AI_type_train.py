#!/usr/bin/env python3

###################
# Argument parsing
###################
import argparse
parser = argparse.ArgumentParser(description = 'Train neural net on organoid data. Outputs weights file.')
parser.add_argument('--train_folder',default = 'single_organoids_train',type=str,help='Folder with training data. Must contain train/categories/pictures and validation/categories/pictures.')
parser.add_argument('--out_folder',default='AI_train_results',type=str,help='Folder to store output in')
args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

def find_n_files_in_folder(folder):
    n_files = 0
    for root,dirs,fnames in os.walk(folder):
        for item in fnames: 
            n_files = n_files + 1
    return(n_files)

os.makedirs(args.out_folder,exist_ok=True)

# dimensions of our images.
img_width, img_height = 120, 120

train_data_dir = os.path.join(args.train_folder,"train")
validation_data_dir = os.path.join(args.train_folder,"validation")

train_classes = os.listdir(train_data_dir)
validation_classes = os.listdir(validation_data_dir)
print("train classes: "+str(train_classes))
print("validation classes"+str(validation_classes))

class_string = None
for s in train_classes:
    if class_string is None:
        class_string = s
    else: 
        class_string = class_string + "\n" +s

with open(os.path.join(args.out_folder,'classes.txt'),'w') as f: 
    f.write(class_string)

nb_train_samples = find_n_files_in_folder(train_data_dir)
nb_validation_samples = find_n_files_in_folder(validation_data_dir)

print("n_train_samples: "+str(nb_train_samples))
print("n_validation_samples: "+str(nb_validation_samples))


epochs = 100
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(len(train_classes)))
model.add(Activation('softmax'))

from keras.optimizers import Adam
o=Adam(lr=0.0005)
model.compile(loss='binary_crossentropy',
              optimizer=o,
              metrics=['binary_accuracy'])

model.summary()

"""
model densenet121 
adam(lr=default)
girhub.com/matkir/master_thesis   TL/ Classifier
"""
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    brightness_range=(0.5,1),
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')


"""
tensorboard
keras.callback.earlystopping(patience=5)
"""
history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
        #callbacks
model.save(os.path.join(args.out_folder,'last_model.h5'))


##################################
# plot run stats
##################################
print(history.history.keys())
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'b',label="Training accuracy")
plt.plot(epochs,val_acc,'r',label="Validation accuracy")
plt.title('Training and Validation accuracy')
plt.legend()
plt.savefig(os.path.join(args.out_folder,"accuracy.png"),dpi=600)

plt.figure()
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()

plt.savefig(os.path.join(args.out_folder,"loss.png"),dpi=600)


