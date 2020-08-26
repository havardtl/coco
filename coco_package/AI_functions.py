import os
import glob
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime

VERBOSE = False

def set_verbose():
    global VERBOSE
    VERBOSE = True

class Training_data:
    def __init__(self,training_dir,validation_dir,validation_n,image_extension = ".png"):
        '''
        Information about where validation and training data is and functions for checking that it is correct.
        
        Params
        training_data_dir   : str : path to training data
        validation_data_dir : str : path to validation data
        validation_n        : int : Number of validation images per category to take from training data
        image_extension     : str : Extension of image files
        '''
        
        self.validation_dir = validation_dir
        self.training_dir = training_dir
        self.validation_n = validation_n
        
        if not os.path.exists(self.training_dir): 
            raise ValueError("Could not find training data folder: "+self.training_dir)
        
        self.image_extension = image_extension
        
        if VERBOSE: 
            print("Organizing training data and validation data for training AI.")
            print("Aiming for "+str(self.validation_n)+" "+self.image_extension+" validation files per category.")
            print("Initial state of training and validation folder: ")
            self.print_folder_stats()
        
        categories_ok,amount_ok = self.check_validation()
        
        if (not categories_ok) or (not amount_ok):
            if VERBOSE: print("Validation folder not ok. So re-making it.")
            self.remove_validation()
            self.make_validation_folder()
            if VERBOSE: 
                print("How it looks after re-organizing")
                self.print_folder_stats()
    
    @classmethod
    def get_subfolders(self,folder): 
        '''
        Get a list of all sub-folders in a folder
        
        Returns 
        sub_folders : list of str : list of all sub_folders in folder
        '''
        sub_folders = []
        for f in os.listdir(folder):
            if os.path.isdir(os.path.join(folder,f)):
                sub_folders.append(f)
        return sub_folders

    def check_validation(self):
        '''
        See if categories in training folder are found in validation folder and has the correct amount of files
        
        Returns
        categories_ok : bool : True if categories are equal for validation and training
        amount_ok     : bool : True if all categories are equal for validation and training and amounts in validation are correct
        '''
        training_categories = self.get_subfolders(self.training_dir)
        
        if len(training_categories) == 0:
            raise ValueError("Did not find any training categories in: "+self.training_dir)
        
        categories_ok = True
        amount_ok = True
        for c in training_categories: 
            val_path = os.path.join(self.validation_dir,c)
            if not os.path.exists(val_path):
                categories_ok = False   
                amount_ok = False
                return categories_ok,amount_ok
            
            image_glob = os.path.join(val_path,"*"+self.image_extension)
            n_images = len(glob.glob(image_glob))
            #if VERBOSE: print(image_glob+"\t"+str(n_images))
            
            if n_images != self.validation_n: 
                amount_ok = False 
        
        if VERBOSE: print("Checked validation folder. Categories ok: "+str(categories_ok)+"\tAmount ok: "+str(amount_ok))
        
        return categories_ok,amount_ok   
    
    def remove_validation(self,require_perfect = True):
        '''
        Move all files from validation folder into training folder
        
        Params
        require_perfect : bool : If true, raise error if some validation categories are not found in training. Else, those files are removed.
        '''
        if not os.path.exists(self.validation_dir):
            if VERBOSE: print("Validation folder does not exist. Not removing it. Validation path: "+self.validation_dir)
            return None
            
        if VERBOSE: print("Moving all files from validation_dir: "+self.validation_dir+" into training_dir: "+self.training_dir)
        
        validation_categories = self.get_subfolders(self.validation_dir)
        
        for c in validation_categories: 
            val_path = os.path.join(self.validation_dir,c)
            training_path = os.path.join(self.training_dir,c)
            
            if os.path.exists(training_path): 
                files = os.listdir(val_path)
                for f in files: 
                    os.rename(os.path.join(val_path,f),os.path.join(training_path,f))
            else: 
                if require_perfect: 
                    print("Require perfect is set to True.")
                    raise ValueError("Categories in validation not found in training! category:"+c)
            
            shutil.rmtree(val_path)
        shutil.rmtree(self.validation_dir)
            
    def make_validation_folder(self):
        '''
        Take a random sample of training files and move it to validation folder
        '''
        if VERBOSE: print("Moving files from training folder to validation folder. validation_n: "+str(self.validation_n))
        os.makedirs(self.validation_dir)
        
        training_categories = self.get_subfolders(self.training_dir)
        
        for c in training_categories:
            training_path = os.path.join(self.training_dir,c)
            val_path = os.path.join(self.validation_dir,c)

            os.makedirs(val_path)
            images_glob = os.path.join(training_path,"*"+self.image_extension)
            #if VERBOSE: print(images_glob)
            image_files = glob.glob(images_glob)
            
            if (len(image_files)<(self.validation_n*2)):
                raise ValueError("Taking out images for validation would give less training images than validation images. validation_n: "+str(self.validation_n)+" images_n: "+str(len(image_files)))
            
            subsample = random.sample(image_files,self.validation_n)
            for f in subsample: 
                fname = os.path.split(f)[1]
                os.rename(os.path.join(training_path,fname),os.path.join(val_path,fname))
    
    def print_folder_stats(self):
        '''
        Print the current number of files in training and validation, not caring about file extensions.
        '''
        def print_dir(dir):
            for root, dirs, files in os.walk(dir, topdown=True):
                n_files = len(files)
                tab_level = root.count("/")*"\t"
                print(tab_level+root+"\t n_files: "+str(n_files))
        
        print("Training folder:")
        print_dir(self.training_dir)
        
        print("Validation folder:")
        print_dir(self.validation_dir)

class AI: 

    def __init__(self,main_folder=None):
        '''
        Bare bones AI class with mainly metadata about AI
        
        Params
        main_folder : str : path to AI folder. If set to None, uses default path in repository 
        '''
        if main_folder is None: 
            this_script_path = os.path.dirname(os.path.realpath(__file__))
            repository_path  = this_script_path.rsplit("/",1)[0]
            self.main_folder = os.path.join(repository_path,"config","AI_train_results")
        else: 
            self.main_folder = main_folder
        
        self.classes_path = os.path.join(self.main_folder,"classes.txt")
        self.training_log_path = os.path.join(self.main_folder,"training_log.txt")
        self.model_path = os.path.join(self.main_folder,"AI_model.h5")
        
        self.classes = None
        
    def read_classes(self):
        '''
        Get classes that AI predicts from file in main AI folder
        '''
        
        with open(self.classes_path,'r') as f: 
            classes = f.readlines()
            
        if len(classes) == 0:
            raise ValueError("Did not find any classes in file: "+classes_path)
            
        for i in range(0,len(classes)):
            classes[i] = classes[i].strip()
            
        self.classes = classes
        
    def get_classes(self): 
        if self.classes is None: 
            self.read_classes()
            
        return self.classes
    
class AI_predict: 
    
    def __init__(self,ai):
        '''
        AI class for predicting new images with pre-trained model
        
        Params
        ai : AI : Bare bones information for AI 
        '''
        
        global K, load_model
        from keras import backend as K
        from keras.models import load_model
        self.ai = ai
        
        self.read_model()
        
    def read_model(self,path = None): 
        '''
        Read pre-trained neural network model used to predict types
        
        Params
        path : str : path to .h5 file with pre-trained keras model. If set to None, loads model from repository
        '''
        from keras.models import load_model
        
        if VERBOSE: 
            print("Preparing to read AI model")
            print("Printing information about AI-model from log-file from AI training.")
            print("Check that this info corresponds to your installation of KERAS, or the model could break. F.ex. if the image_data_format does not correspond to your installation.")
            if not os.path.exists(self.ai.training_log_path):
                print("WARNING: Could not find AI model file: "+self.ai.training_log_path+". So can't print out the running info.")
            else: 
                with open(self.ai.training_log_path,'r') as f: 
                    training_info = f.readlines()
                training_info = training_info[:2]
                for l in training_info: 
                    print(l)
                print("\n")
        
        if path is None:
            path = self.ai.model_path
            
        print("Loading model from: "+path)
        self.model = load_model(path)
        
    def get_predictions(self,objects): 
        '''
        Predict the type of object from images of it 
        
        Params
        objects           : list of np.array : List of grayscale images to predict
        
        Returns 
        predictions : list of str : Predictions 
        '''
        
        if VERBOSE: print("Predicting objects with AI")
        
        objects_correct_format = []
        
        keras_data_format = K.image_data_format()
        for o in objects: 
            if keras_data_format == "channels_first": 
                o = o[np.newaxis,...]
            elif keras_data_format == "channels_last": 
                o = o[...,np.newaxis]
            objects_correct_format.append(o) 
        
        objects_correct_format = np.array(objects_correct_format)
        
        model_predictions = self.model.predict(objects_correct_format)
        
        classes = self.ai.get_classes()
        predictions = []
        for p in model_predictions: 
            type = classes[np.argmax(p)]
            predictions.append(type)
        
        return predictions


class AI_train:
    
    def __init__(self,ai,training_data,overwrite=False):
        '''
        AI class used for training new model with pre-classified images
        
        Params
        ai                  : AI            : Bare bones information for AI 
        training_data       : Training_data : Class with references to training and validation data that is checked to be correct.
        overwrite           : bool          : If True, overwrites existing AI folder. 
        '''
        global ImageDataGenerator, Sequential, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, K, Adam
        
        from keras.preprocessing.image import ImageDataGenerator
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D
        from keras.layers import Activation, Dropout, Flatten, Dense
        from keras import backend as K
        from keras.optimizers import Adam
        
        self.ai = ai
        self.train_data_dir = training_data.training_dir
        self.validation_data_dir = training_data.validation_dir
        
        if os.path.exists(self.ai.main_folder): 
            if overwrite: 
                print("Main AI folder already exists. Deleting: "+self.ai.main_folder)
                shutil.rmtree(self.ai.main_folder)
            else: 
                raise RuntimeError("AI folder already exists and you have not set it to overwrite. Try --help for options.")
        
        if VERBOSE: print("Making new AI folder: "+self.ai.main_folder)
        os.makedirs(self.ai.main_folder)
        
        self.epochs = 1 #100
        self.batch_size = 128
        self.img_width, self.img_height = 120, 120
        
        self.start_time = datetime.datetime.now()
        self.log_str = ""
        self.add_to_log("Training KERAS AI")
        self.add_to_log("Time: "+self.start_time.strftime("%d/%m/%Y %H:%M:%S"))
        self.add_to_log("image_data_format = "+K.image_data_format())
        self.add_to_log("ai.main_folder = "+self.ai.main_folder)
        self.add_to_log("train_data_dir = "+self.train_data_dir)
        self.add_to_log("validation_data_dir = "+self.validation_data_dir)
        self.add_to_log("img_width: "+str(self.img_width)+"\timg_height: "+str(self.img_height))
        self.add_to_log("epochs = "+str(self.epochs))
        self.add_to_log("batch_size = "+str(self.batch_size))
        
        self.get_training_classes()
        self.build_keras_model()
        self.train_model()
        self.write_log()
        
    @classmethod
    def find_n_files_in_folder(self,folder):
        n_files = 0
        for root,dirs,fnames in os.walk(folder):
            for item in fnames: 
                n_files = n_files + 1
        return(n_files) 
    
    def add_to_log(self,string):
        '''
        Write string to end of log-file
        
        Params
        string    : str  : Add string to log_str with newline
        '''
        if VERBOSE: print(string)
        self.log_str += string + "\n"
        
    def write_log(self):
        # Write log string to log
        if VERBOSE: print("Write log to: "+self.ai.training_log_path)
        with open(self.ai.training_log_path,'w') as f:
            f.write(self.log_str)
    
    def get_training_classes(self):
        '''
        Read training and validation classes and number of images used for training of model and write them to file
        '''
        def get_folder_and_n_files(path):
            '''
            Get names of folders in sub-folder and files in it
            '''
            folders = []
            n_files = []
            for f in os.listdir(path):
                folders.append(f)
                n_files.append(self.find_n_files_in_folder(os.path.join(path,f)))
                
            return folders,n_files
        
        self.add_to_log("Getting training and validation classes")
        
        self.add_to_log("Finding training samples in: "+self.train_data_dir)
        train_classes,train_classes_n = get_folder_and_n_files(self.train_data_dir)
        self.nb_train_samples = sum(train_classes_n)
        
        self.add_to_log("Training images: "+str(self.nb_train_samples))
        for c,n in zip(train_classes,train_classes_n):
            self.add_to_log("\t"+c+"\t"+str(n))
            
        if self.nb_train_samples < 1: 
            raise ValueError("Did not find any training samples!")
        
        self.add_to_log("Finding validation samples in: "+self.validation_data_dir)
        validation_classes,validation_classes_n = get_folder_and_n_files(self.validation_data_dir)
        self.nb_validation_samples = sum(validation_classes_n)
        
        self.add_to_log("Validation images: "+str(self.nb_validation_samples))
        for c,n in zip(validation_classes,validation_classes_n):
            self.add_to_log("\t"+c+"\t"+str(n))
        
        if self.nb_validation_samples < 1: 
            raise ValueError("Did not find any validation samples!")
        
        if len(train_classes) <2: 
            raise ValueError("Found less than two training classes. See above.")
        
        if len(validation_classes) <2: 
            raise ValueError("Found less than two validation classes. See above.")
            
        if len(validation_classes) != len(train_classes): 
            raise ValueError("Validation and training classes not equal length. See above")
        
        class_string = "\n".join(train_classes)
        
        self.add_to_log("Writing classes to file: "+self.ai.classes_path)
        with open(self.ai.classes_path,'w') as f: 
            f.write(class_string)
            
        self.train_classes = train_classes
            
    def build_keras_model(self):
        '''
        Build keras model 
        '''
        self.add_to_log("Building KERAS model")
        
        if K.image_data_format() == 'channels_first':
            input_shape = (1, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, 1)

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
        model.add(Dense(len(self.train_classes)))
        model.add(Activation('softmax'))

        o=Adam(lr=0.0005)
        model.compile(loss='binary_crossentropy',
                      optimizer=o,
                      metrics=['binary_accuracy'])

        model.summary()
        
        #self.add_to_log(model_out)
        
        self.model = model
        
    def train_model(self):
        '''
        Train model
        '''
        self.add_to_log("Training model")
        
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            brightness_range=(0.5,1),
            rotation_range=360,
            horizontal_flip=True,
            vertical_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='grayscale')

        validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='grayscale')

        self.history = self.model.fit_generator(
            train_generator,
            steps_per_epoch= self.nb_train_samples // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=self.nb_validation_samples // self.batch_size)
            
        self.model.save(self.ai.model_path)
    
    def plot_run_stats(self):
        '''
        Plot statistics from run
        '''
        self.add_to_log("Plotting running stats")
        
        #print("Available stats: "+str(history.history.keys()))
        acc = history.history['binary_accuracy']
        val_acc = history.history['val_binary_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1,len(acc)+1)
        
        accuracy_plot_path = os.path.join(self.ai.main_folder,"accuracy.png")
        self.add_to_log("Plotting Accuracy to file: "+accuracy_plot_path)
        plt.plot(epochs,acc,'b',label="Training accuracy")
        plt.plot(epochs,val_acc,'r',label="Validation accuracy")
        plt.title('Training and Validation accuracy')
        plt.legend()
        plt.savefig(accuracy_plot_path,dpi=600)
        
        loss_plot_path = os.path.join(self.ai.main_folder,"loss.png")
        self.add_to_log("Plotting loss to file: "+loss_plot_path)
        plt.figure()
        plt.plot(epochs,loss,'b',label='Training loss')
        plt.plot(epochs,val_loss,'r',label="Validation loss")
        plt.title("Training and Validation loss")
        plt.legend()

        plt.savefig(loss_plot_path,dpi=600)




















