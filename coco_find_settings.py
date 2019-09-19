#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Process one of the confocal images with a multitude of segmentation settings to find the best settings. ')
parser.add_argument('--raw_data',default="rawdata",type=str,help='Path to raw image. If a folder is submitted the alfabetically first is choosen. Can use all formats accepted by your installed version of bftools. ')
parser.add_argument('--results',type=str,default = 'segment_test', help='Output folder for segmentation test results')
parser.add_argument('--config_file',type = int,default=None,help='Output config file for parameters to test. If no input a standard set is tested.')
parser.add_argument('--z_toshow',type = str,default="i3",help='z-slices to include. "1:3" = [1,2,3],"1,5,8" = [1,5,8] and "i3" = 3 evenly choosen from range. Default = "i3"')
parser.add_argument('--temp_folder',type=str,default='coco_temp',help="temp folder for storing temporary images. Must not exist before startup. default: ./coco_temp")
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores minus 1.')
args = parser.parse_args()

########################
# Imports 
########################
import os

#if os.path.isdir(args.temp_folder):
#    raise ValueError('--temp_folder already exists. This one is deleted by the program and must be clean')
import math
import time
import datetime
import shutil
import random
import multiprocessing as mp 

import cv2
import numpy as np 
import pandas as pd

import utilities.general_functions as oc
import utilities.image_processing_functions as oi

def choose_z_slices(all_z_slices,to_choose):
    '''
    Extract only some of the z-slices
    
    Params
    all_z_slices : list of str : All available z-slices
    to_choose    : str         : which z_slice to choose. "1:3" = [1,2,3],"1,5,8" = [1,5,8] and "i3" = 3 evenly choosen from range.

    Return
    choosen_z_slices : list of str : z_slices to process
    '''
    choosen_z_slices = []
    if "i" in to_choose:
        i_to_keep = int(args.z_toshow.replace("i",""))
        step_size = math.ceil(len(all_z_slices)/i_to_keep)

        for i in range(0,len(all_z_slices),step_size):
            choosen_z_slices.append("Z"+str(i))
    elif "," in to_choose:
        temp = args.split(",")
        for i in range(len(temp)):
            choosen_z_slices.append("Z"+str(int(temp[i])))
    else: 
        choosen_z_slices.append("Z"+int(to_choose))

    return choosen_z_slices

##############################
# Run program
##############################
os.makedirs(args.results,exist_ok=True)
os.makedirs(args.temp_folder,exist_ok = True)

temp_mask_folder = os.path.join(args.temp_folder,"masks")
os.makedirs(temp_mask_folder,exist_ok=True)

if os.path.isdir(args.raw_data):
    image_path_all = os.listdir(args.raw_data)
    image_path_all.sort()
    image_path = os.path.join(args.raw_data,image_path_all[0])
else: 
    image_path = args.raw_data

image_id,extension = os.path.splitext(image_path)
folder,image_id = os.path.split(image_id)

print("Processing image {image}".format(image = image_id))

#images_paths = oi.get_images(image_path,args.temp_folder)
#images_paths.to_csv(os.path.join(args.temp_folder,"images_path.csv"))
images_paths = pd.read_csv(os.path.join(args.temp_folder,"images_path.csv"),index_col = 0)

choosen_z_slices = choose_z_slices(images_paths["Z_index"].unique(),args.z_toshow)
images_paths = images_paths[images_paths["Z_index"].isin(choosen_z_slices)]

full_path = []
for i in images_paths.index:
    full_path.append(os.path.join(images_paths.loc[i,"root"],images_paths.loc[i,"file"]))
images_paths["full_path"] = full_path

test_settings = oi.make_test_settings_df(images_paths["full_path"],temp_mask_folder)
test_settings = test_settings.merge(images_paths.drop(["root","file","info"],axis='columns'),left_on="image_path",right_on="full_path")

test_settings.to_csv(os.path.join(args.temp_folder,"test_settings.csv"))

if args.cores is None:
    cores = mp.cpu_count()-1
else:
    cores = args.cores

tot_test_images = len(test_settings.index)
image_info = []
if cores==1: 
    for i in test_settings.index:
        info = str(i)+"/"+str(tot_test_images)
        image_info.append(oi.get_processed_mask(test_settings.loc[i,"image_path"],info,test_settings.loc[i,"shrink"],test_settings.loc[i,"contrast"],test_settings.loc[i,"auto_max"],test_settings.loc[i,"thresh_type"],test_settings.loc[i,"thresh_upper"],test_settings.loc[i,"thresh_lower"],test_settings.loc[i,"open_kernel"],test_settings.loc[i,"close_kernel"],test_settings.loc[i,"mask_path"]))

else: 
    pool = mp.Pool(cores)

    for i in test_settings.index: 
        info = str(i)+"/"+str(tot_test_images)
        image_info.append(pool.apply_async(oi.get_processed_mask,args=(test_settings.loc[i,"image_path"],info,test_settings.loc[i,"shrink"],test_settings.loc[i,"contrast"],test_settings.loc[i,"auto_max"],test_settings.loc[i,"thresh_type"],test_settings.loc[i,"thresh_upper"],test_settings.loc[i,"thresh_lower"],test_settings.loc[i,"open_kernel"],test_settings.loc[i,"close_kernel"],test_settings.loc[i,"mask_path"])))

    pool.close()
    pool.join()

    image_info = [x.get() for x in image_info]

df_plot = oi.plot_images_pdf(test_settings,["channel_index","series_index","T_index"],["Z_index"],["processed","auto_max","shrink","thresh_type","thresh_upper","thresh_lower","contrast","open_kernel","close_kernel"],(512,512))


