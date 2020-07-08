#!/usr/bin/env python3

###################
# Argument parsing
###################
import argparse
parser = argparse.ArgumentParser(description = 'Train neural net on organoid data. Outputs weights file.')
parser.add_argument('--pictures_folder',default = 'single_objects',type=str,help='Folder with training data. Must contain train/<category>/<pictures> and potentially validation/<category>/<pictures>. ')
parser.add_argument('--n_validation',default=500,type=int,help='Number of files to take out from training samples per category for validation of neural network.')
parser.add_argument('--out_folder',default="AI_train_results",type=str,help='Folder to store output in. Default is to make a new folder in current directory.')
parser.add_argument('--image_extension',default=".png",type=str,help='Extension of image files to use in training.')
parser.add_argument('--overwrite',default=False,action="store_true",help="Overwrite existing --out_folder, as that folder needs to be empty.")
parser.add_argument('--silent',default=False,action="store_true",help="Do not print information about AI run.")
args = parser.parse_args()

import os
import classes.AI_functions as ai

if not args.silent: 
    ai.VERBOSE = True 

train_data_dir = os.path.join(args.pictures_folder,"train")
validation_data_dir = os.path.join(args.pictures_folder,"validation")

training_data = ai.Training_data(train_data_dir,validation_data_dir,args.n_validation,args.image_extension)

train_ai = ai.AI_train(ai.AI(args.out_folder),training_data,overwrite=args.overwrite)


