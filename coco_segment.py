#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Segment objects in confocal images and extracts their information.')
parser.add_argument('--raw_folder',default="raw/rawdata",type=str,help='Folder with all raw data. Does not look for files in sub-folders. Can use all formats accepted by your installed version of bftools.')
parser.add_argument('--raw_file',type=str,help='Supply a single raw file to process. Ignores --raw_folder if supplied.')
parser.add_argument('--annotation_file',type=str,default='annotation1.xlsx',help="Excel file with segmentation settings for channels. Make it with coco_make_annotation_file.py.")
parser.add_argument('--out_contours',type=str,default = 'stats/contours_stats', help='Output folder for stats about contours')
parser.add_argument('--out_rois',type=str,default = 'stats/rois_stats', help='Output folder for stats about 3d rois')
parser.add_argument('--out_graphical',type=str,default = 'graphical/graphic_segmentation', help='Output folder for graphical representation of segmentation')
parser.add_argument('--temp_folder',type=str,default='raw/temp',help="temp folder for storing temporary files.")
parser.add_argument('--extracted_images_folder',type=str,default='raw/extracted_raw',help="Folder with extracted images.")
parser.add_argument('--stitch',default=False,action="store_true",help='Switch that turns on stitching of images using ImageJ. NB! Very experimental')
parser.add_argument('--cores',type=int,default=1,help='Number of cores to use. Default is 1. -1 = number of cores -1')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted files')
parser.add_argument('--debug',action='store_true',default=False,help='Run in verbose mode and with one core for debugging')
parser.add_argument('--verbose',action='store_true',default=False,help='Verbose mode')
parser.add_argument('--overwrite',action = 'store_true',default=False,help='Overwrite all files rather than re-using exisiting files. Does not re-extract images. NB! does this by deleting --temp_folder, --out_rois, --out_graphical, --out_contours and all their content')
args = parser.parse_args()

########################
# Imports 
########################
import os
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
if not os.path.exists(args.annotation_file): 
    cmd = "coco_make_annotation_file.py"
    exit_code = os.system(cmd)
    if exit_code != 0: 
        raise RunTimeError("Command did not finish properly. cmd: "+cmd)

if args.overwrite:
    for folder in [args.temp_folder,args.out_rois,args.out_graphical,args.out_contours]:
        oi.delete_folder_with_content(folder)

os.makedirs(args.out_graphical,exist_ok=True)
os.makedirs(args.out_contours,exist_ok=True)
os.makedirs(args.out_rois,exist_ok=True)
os.makedirs(args.temp_folder,exist_ok = True)
os.makedirs(args.extracted_images_folder,exist_ok = True)

pickle_folder = os.path.join(args.temp_folder,"segment_pickles")
os.makedirs(pickle_folder,exist_ok=True)

if args.raw_file is None: 
    raw_imgs = []
    for i in os.listdir(args.raw_folder): 
        raw_path = os.path.join(args.raw_folder,i)
        img_i = classes.Image_info(raw_path,args.temp_folder,args.extracted_images_folder,pickle_folder)
        raw_imgs.append(img_i)
    raw_imgs.sort()
else: 
    raw_imgs = [classes.Image_info(args.raw_file,args.temp_folder,args.extracted_images_folder,pickle_folder)]
    args.cores = 1

if not args.n_process is None:
    raw_imgs = raw_imgs[0:args.n_process]

if args.debug: 
    args.cores = 1
    args.verbose = True

classes.VERBOSE = args.verbose
classes.CONTOURS_STATS_FOLDER = args.out_contours 
classes.TEMP_FOLDER = args.temp_folder 
classes.GRAPHICAL_SEGMENTATION_FOLDER = args.out_graphical

print("Found {n_files} to process".format(n_files = len(raw_imgs)))
segment_settings = oi.excel_to_segment_settings(args.annotation_file)

if args.stitch: 
    for image_info in raw_imgs: 
        image_info.get_extracted_files_path(extract_with_imagej=args.stitch)

def main(image_info,segment_settings,info):
    '''
    Process one file of the program 

    Params
    image_info       : Image_info               : Information about file to process
    segment_settings : list of Segment_settings : Information about how to process images
    info             : str                      : Info about the image that is processed to be printed
    '''
    
    global args
    df_rois_path = os.path.join(args.out_rois,image_info.file_id+".csv")
    
    print_str = str(info)+"\traw_img_path: "+str(image_info.raw_path)+"\t"
    
    if os.path.exists(df_rois_path): 
        print(print_str+"3D ROIs already made, skipping this file")
    else:
        print(print_str+"Making 3D rois")
        if not os.path.isfile(image_info.pickle_path): 
            images_paths = image_info.get_extracted_files_path(extract_with_imagej=args.stitch)
            z_stacks = oi.img_paths_to_zstack_classes(images_paths,segment_settings)
            for z in z_stacks:
                z.make_masks()
                z.make_combined_masks()
                z.find_contours()
                z.group_contours()
                z.is_inside_combined()
                z.find_z_overlapping()
                z.update_contour_stats()
                z.measure_channels()
                z.write_contour_info()
                
                if args.verbose: z.print_all()
                
                try: 
                    with open(image_info.pickle_path, 'wb') as handle:
                        if args.verbose: print("Writing pickle object")
                        pickle.dump(z_stacks, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except: 
                    os.remove(image_info.pickle_path)
                    print("Could not write pickle object")
        else: 
            if args.verbose: print("Reading z_stacks info from pickle file: "+image_info.pickle_path)
            with open(image_info.pickle_path, 'rb') as handle:
                z_stacks = pickle.load(handle)
        
        rois_3d = []
        for z in z_stacks: 
            temp_rois_3d = z.get_rois_3d()
            for roi in temp_rois_3d: 
                rois_3d.append(roi)
        
        df_rois = pd.DataFrame()
        if args.verbose: print("Building 3D rois")
        for i in range(len(rois_3d)): 
            if args.verbose: oi.print_percentage(i,len(rois_3d),10)
            rois_3d[i].build()
            df_rois = df_rois.append(rois_3d[i].data,ignore_index=True,sort=False)
        
        if args.verbose: 
            print("Writing df_rois to path: "+df_rois_path)
            print(df_rois)
        df_rois.to_csv(df_rois_path)
        
        for z in z_stacks: 
            z.to_pdf()
    return None

if args.cores is -1:
    args.cores = mp.cpu_count()-1

tot_images = len(raw_imgs)

image_info = []
if args.cores==1: 
    for i in range(0,len(raw_imgs)):
        info = str(i+1)+"/"+str(tot_images)
        image_info.append(main(raw_imgs[i],segment_settings,info))
else: 
    pool = mp.Pool(args.cores)

    for i in range(0,len(raw_imgs)):
        info = str(i+1)+"/"+str(tot_images)
        image_info.append(pool.apply_async(main,args=(raw_imgs[i],segment_settings,info)))

    pool.close()
    pool.join()

    image_info = [x.get() for x in image_info]

