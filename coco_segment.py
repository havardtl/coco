#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Segment objects in confocal images and extracts their information.')
parser.add_argument('--raw_data',default="rawdata",type=str,help='Folder with all raw data. Can use all formats accepted by your installed version of bftools. ')
parser.add_argument('--out_projections',type=str,default = 'min_projections', help='Output folder for minimal projections')
parser.add_argument('--out_annotations',type=str,default = 'annotations', help='Output folder for annotations')
parser.add_argument('--out_segmented',type=str,default = 'graphic_out_segment_raw', help='Output folder for graphical segmentation')
parser.add_argument('--segment_settings',type=str,default = 'segment_settings.xlsx', help='Settings for each channels segmentation')
parser.add_argument('--temp_folder',type=str,default='coco_temp',help="temp folder for storing temporary images. Must not exist before startup. default: ./ORGcount_temp")
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores minus 1.')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted files')
parser.add_argument('--overwrite',action = 'store_true',help='Re-process images that have annotations.')
args = parser.parse_args()


########################
# Imports 
########################
import os

#if os.path.isdir(args.temp_folder):
#    raise ValueError('--temp_folder already exists. This one is deleted by the program and must be clean')

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

##############################
# Run program
##############################
os.makedirs(args.out_projections,exist_ok=True)
os.makedirs(args.out_segmented,exist_ok=True)
os.makedirs(args.out_annotations,exist_ok=True)
os.makedirs(args.temp_folder,exist_ok = True)

extracted_raw_folder = os.path.join(args.temp_folder,"extracted_raw")

os.makedirs(extracted_raw_folder,exist_ok = True)

df_all = pd.DataFrame()

image_ids_all = os.listdir(args.raw_data)
for i in range(len(image_ids_all)): 
    image_ids_all[i] = os.path.join(args.raw_data,image_ids_all[i])

if not args.n_process is None:
    image_ids_all.sort()
    image_ids_all = image_ids_all[0:args.n_process]

print("Found {n_files} to process".format(n_files = len(image_ids_all)))

segment_settings = pd.read_excel(args.segment_settings)
open_kernel = []
close_kernel = []
for i in segment_settings.index:
    open_val = segment_settings.loc[i,"open_kernel"]
    open_kernel.append(np.ones((open_val,open_val),np.uint8))
    close_val = segment_settings.loc[i,"close_kernel"]
    close_kernel.append(np.ones((close_val,close_val),np.uint8))

segment_settings["open_kernel"] = open_kernel
segment_settings["close_kernel"] = close_kernel 


def main(raw_img_path,info,temp_folder,segment_settings):
    '''
    Process one file of the program 

    Params
    raw_img_path     : str          : Path to microscopy image file to process
    info             : str          : Info about the image that is processed to be printed
    segment_settings : pd.DataFrame : Data frame with segmentation settings. rows = channels, columns = setting. See oi.get_processed_mask for settings needed.
    '''
    img_id = os.path.splitext(os.path.split(raw_img_path)[1])[0]

    mask_save_folder = os.path.join(temp_folder,"masks",img_id)
    os.makedirs(mask_save_folder,exist_ok=True)
    images_raw_extracted_folder = os.path.join(temp_folder,"extracted_raw",img_id)
    df_rois_path = os.path.join(temp_folder,"df_rois",img_id+".pkl")
    xml_folder = os.path.join(temp_folder,"xml_files")
    os.makedirs(xml_folder,exist_ok=True)

    if os.path.exists(df_rois_path): 
        df_rois = pd.read_pickle(df_rois_path)
    else:
        images_raw_extracted_infofile = os.path.join(images_raw_extracted_folder,"files_info.csv")
        if os.path.isfile(images_raw_extracted_infofile): 
            print(info+"\traw_img_path: "+raw_img_path+"\tImages already extracted, using those.")
            df_images = pd.read_csv(images_raw_extracted_infofile,index_col = 0)
        else:
            print(info+"\traw_img_path: "+raw_img_path+"\tExtracting images...")
            df_images = oi.get_images(raw_img_path,images_raw_extracted_folder)
        
        df_images = oi.add_all_channels(df_images)
        
        df_images["mask_path"] = None
        for i in df_images.index:
            df_images.loc[i,'mask_path'] = oi.make_mask(df_images.loc[i,],segment_settings,mask_save_folder)
        
        df_images = df_images.merge(segment_settings[['channel_index','shrink']],on="channel_index",how="left")
        df_rois = pd.DataFrame() 
        for i in df_images.index:
            temp = oi.get_rois(df_images.loc[i,])
            df_rois = df_rois.append(temp,ignore_index=True)
        
        df_rois["z_int"] = df_rois["Z_index"].str.replace("Z","").astype(int)
        df_rois["z_stack_id"] = df_rois['series_index']+df_rois['T_index']+df_rois['channel_index']
        
        df_rois["overlapping_z"] = None
        df_rois["x_res_um"] = None
        df_rois["y_res_um"] = None
        df_rois["z_res_um"] = None
        for z_stack_id in df_rois['z_stack_id'].unique():
            z_max = df_rois.loc[df_rois["z_stack_id"]==z_stack_id,"z_int"].max()
            for i in range(z_max):
                this_z = (df_rois['z_stack_id']==z_stack_id) & (df_rois['z_int']==i)
                next_z = (df_rois['z_stack_id']==z_stack_id) & (df_rois['z_int']== (i+1))
                if sum(this_z) > 0:
                    if sum(next_z) > 0:
                        print(df_rois.loc[this_z,"overlapping_z"])
                        df_rois.loc[this_z,"overlapping_z"] = oi.get_overlapping_contours(df_rois[this_z],df_rois[next_z])
            
            z_stack = df_rois[df_rois["z_stack_id"]==z_stack_id]
            ome_tiff_path = os.path.join(z_stack["root"].iloc[0],z_stack["file"].iloc[0])
            
            x_res,y_res,z_res = oi.get_xyz_res(ome_tiff_path,xml_folder)
            df_rois.loc[df_rois["z_stack_id"]==z_stack_id,"x_res_um"] = x_res
            df_rois.loc[df_rois["z_stack_id"]==z_stack_id,"y_res_um"] = y_res
            df_rois.loc[df_rois["z_stack_id"]==z_stack_id,"z_res_um"] = z_res
        
        channels = df_rois["channel_index"].unique()
        for i in channels:
            df_rois["is_inside_"+str(i)] = None
        
        for i in df_rois.index: 
            this_image = df_rois.loc[df_rois["info_no_channel"]==df_rois.loc[i,"info_no_channel"],]
            for c in channels: 
                if this_image.loc[i,"channel_index"] == c: 
                    df_rois.loc[i,"is_inside_"+str(c)] = None
                else: 
                    df_rois.loc[i,"is_inside_"+str(c)] = oi.check_if_inside(df_rois.loc[i,],this_image[this_image["channel_index"]==c])
        
        os.makedirs(os.path.split(df_rois_path)[0],exist_ok=True) 
        df_rois.to_pickle(df_rois_path)
        
    objects = oi.build_objects(df_rois)
    
    objects_file_path = os.path.join(args.out_annotations,img_id+".csv") 
    objects.to_csv(objects_file_path)
    
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
        image_info.append(main(image_ids_all[i],info,args.temp_folder,segment_settings))
else: 
    pool = mp.Pool(cores)

    for i in range(0,len(image_ids_all)):
        info = str(i+1)+"/"+str(tot_images)
        image_info.append(pool.apply_async(main,args=(image_ids_all[i],info,args.temp_folder,segment_settings)))

    pool.close()
    pool.join()

    image_info = [x.get() for x in image_info]

