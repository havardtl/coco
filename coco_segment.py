#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Segment objects in confocal images and extracts their information.')
parser.add_argument('--raw_data',default="rawdata",type=str,help='Folder with all raw data. Can use all formats accepted by your installed version of bftools. ')
parser.add_argument('--out_projections',type=str,default = 'min_projections', help='Output folder for minimal projections')
parser.add_argument('--out_contours',type=str,default = 'contours_stats', help='Output folder for stats about contours')
parser.add_argument('--out_rois',type=str,default = 'rois_stats', help='Output folder for stats about 3d rois')
parser.add_argument('--out_graphical',type=str,default = 'graphic_segmentation', help='Output folder for graphical representation of segmentation')
parser.add_argument('--segment_settings',type=str,default = 'segment_settings.xlsx', help='Settings for each channels segmentation')
parser.add_argument('--temp_folder',type=str,default='coco_temp',help="temp folder for storing temporary images. Must not exist before startup. default: ./ORGcount_temp")
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores minus 1.')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted files')
parser.add_argument('--debug',action='store_true',default=False,help='Run in verbose mode, files alfabetically, and with one core for debugging')
parser.add_argument('--verbose',action='store_true',default=False,help='Verbose mode')
parser.add_argument('--overwrite',action = 'store_true',default=False,help='Overwrite all files rather than re-using exisiting files. NB! does this by deleting --temp_folder and all its content')
args = parser.parse_args()

########################
# Imports 
########################
import os

import time
import datetime
import shutil
import random
import multiprocessing as mp 

import cv2
import numpy as np 
import pandas as pd

import utilities.image_processing_functions as oi
import utilities.classes as classes

import pickle

##############################
# Run program
##############################
if args.overwrite: 
    exit_code = os.system('rm -rf '+args.temp_folder)
    if exit_code != 0: 
        raise ValueError("Could not delete temp_folder: "+args.temp_folder)

os.makedirs(args.out_projections,exist_ok=True)
os.makedirs(args.out_graphical,exist_ok=True)
os.makedirs(args.out_contours,exist_ok=True)
os.makedirs(args.out_rois,exist_ok=True)
os.makedirs(args.temp_folder,exist_ok = True)

extracted_raw_folder = os.path.join(args.temp_folder,"extracted_raw")

os.makedirs(extracted_raw_folder,exist_ok = True)

df_all = pd.DataFrame()

image_ids_all = os.listdir(args.raw_data)
for i in range(len(image_ids_all)): 
    image_ids_all[i] = os.path.join(args.raw_data,image_ids_all[i])

image_ids_all.sort()

if not args.n_process is None:
    image_ids_all = image_ids_all[0:args.n_process]

if args.debug: 
    args.cores = 1
    args.verbose = True

classes.VERBOSE = args.verbose
classes.CONTOURS_STATS_FOLDER = args.out_contours 
classes.TEMP_FOLDER = args.temp_folder 
classes.GRAPHICAL_SEGMENTATION_FOLDER = args.out_graphical

print("Found {n_files} to process".format(n_files = len(image_ids_all)))

df_segment_settings = pd.read_excel(args.segment_settings)

segment_settings = []
for channel in df_segment_settings["channel_index"]: 
    s = df_segment_settings.loc[df_segment_settings["channel_index"]==channel,]
    if len(s["channel_index"])!=1:
        raise ValueError("Need at least one unique channel index in segment settings.")
    s = s.iloc[0,]
    segment_settings_c = classes.Segment_settings(channel,s["color"],s["contrast"],s["auto_max"],s["thresh_type"],s["thresh_upper"],s["thresh_lower"],s["open_kernel"],s["close_kernel"],s["combine"])
    segment_settings.append(segment_settings_c)

def main(raw_img_path,info,segment_settings):
    '''
    Process one file of the program 

    Params
    raw_img_path     : str          : Path to microscopy image file to process
    info             : str          : Info about the image that is processed to be printed
    segment_settings : pd.DataFrame : Data frame with segmentation settings. rows = channels, columns = setting. See oi.get_processed_mask for settings needed.
    '''
    global args

    img_id = os.path.splitext(os.path.split(raw_img_path)[1])[0]

    extracted_images_folder = os.path.join(args.temp_folder,"extracted_raw",img_id)
    df_rois_path = os.path.join(args.out_rois,img_id+".csv")
    
    pickle_folder = os.path.join(args.temp_folder,"pickles")
    os.makedirs(pickle_folder,exist_ok=True)
    pickle_path = os.path.join(pickle_folder,img_id+".pickle")
    
    print_str = info+"\traw_img_path: "+raw_img_path+"\t"
   
    if os.path.exists(df_rois_path): 
        print(print_str+"3D ROIs already made, skipping this file")
    else:
        images_paths_file = "files_info.txt"
        bfconvert_info_str = "_INFO_%s_%t_%z_%c"
        file_ending = ".ome.tiff"

        if os.path.isfile(os.path.join(extracted_images_folder,images_paths_file)): 
            print(print_str+"Images already extracted, using those to build 3D ROIs.")
            with open(os.path.join(extracted_images_folder,images_paths_file),'r') as f: 
                images_paths = f.read().splitlines()
        else:
            print(print_str+"Extracting images and building 3D ROIs")
            images_paths = oi.get_images(raw_img_path,extracted_images_folder,images_paths_file,bfconvert_info_str,file_ending)
        
        if not os.path.isfile(pickle_path): 
            z_stacks = oi.img_paths_to_zstack_classes(images_paths,file_ending,segment_settings)
            for z in z_stacks:
                if args.verbose: 
                    z.print_all()
                z.make_projections()
                z.make_masks()
                z.make_combined_masks()
                z.find_contours()
                z.is_inside_combined()
                z.find_z_overlapping()
                z.update_contour_stats()
                z.measure_channels()
                z.write_contour_info()
                
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(z_stacks, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else: 
            if args.verbose:
                print("Reading z_stacks info from pickle file: "+pickle_path)
            with open(pickle_path, 'rb') as handle:
                z_stacks = pickle.load(handle)
        
        rois_3d = []
        for z in z_stacks: 
            temp_rois_3d = z.get_rois_3d()
            for roi in temp_rois_3d: 
                rois_3d.append(roi)
        
        df_rois = pd.DataFrame()
        for roi in rois_3d: 
            roi.build()
            df_rois = df_rois.append(roi.data,ignore_index=True,sort=False)
        
        if args.verbose: 
            print("Writing df_rois to path: "+df_rois_path)
            print(df_rois)
        df_rois.to_csv(df_rois_path)
        
        for z in z_stacks: 
            z.to_pdf()
    return None

if args.cores is None:
    cores = mp.cpu_count()-1
else:
    cores = args.cores

tot_images = len(image_ids_all)

image_info = []
if cores==1: 
    for i in range(0,len(image_ids_all)):
        info = str(i+1)+"/"+str(tot_images)
        image_info.append(main(image_ids_all[i],info,segment_settings))
else: 
    pool = mp.Pool(cores)

    for i in range(0,len(image_ids_all)):
        info = str(i+1)+"/"+str(tot_images)
        image_info.append(pool.apply_async(main,args=(image_ids_all[i],info,segment_settings)))

    pool.close()
    pool.join()

    image_info = [x.get() for x in image_info]

