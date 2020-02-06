#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Segment objects in confocal images and extracts their information. Makes minimal projection before segmenting. ')
parser.add_argument('--raw_folder',default="raw/rawdata",type=str,help='Folder with all raw data. Does not look for files in sub-folders. Can use all formats accepted by your installed version of bftools.')
parser.add_argument('--raw_file',type=str,help='Supply a single raw file to process. Ignores --raw_folder if supplied.')
parser.add_argument('--settings_file',type=str,default='annotation1.xlsx',help="Excel file with segmentation settings for channels. Make it with coco_make_annotation_file.py.")
parser.add_argument('--annotations',type=str,default='annotations',help="Folder with annotation of channel images")
parser.add_argument('--masks',type=str,default=None,help="Folder with masks. If supplied, z-stacks are split into one z_stack per mask. Naming convention for masks: '{file_id}_*_MASK_{mask_name}.*'. 255 = area to keep. NB! Black holes in masks are ignored, so avoid donut like shapes as masks.")
parser.add_argument('--categories',type=str,help='File to load category information from. Default is to load it from default file in utilities/categories.csv')
parser.add_argument('--out_contours',type=str,default = 'stats/2D_contours_stats', help='Output folder for stats about contours')
parser.add_argument('--out_graphical',type=str,default = 'graphical/2D_graphic_segmentation', help='Output folder for graphical representation of segmentation')
parser.add_argument('--temp_folder',type=str,default='raw/temp',help="temp folder for storing temporary files.")
parser.add_argument('--extracted_images_folder',type=str,default='raw/extracted_raw',help="Folder with extracted images.")
parser.add_argument('--stitch',default=False,action="store_true",help='Switch that turns on stitching of images using ImageJ. NB! Very experimental')
parser.add_argument('--cores',type=int,default=1,help='Number of cores to use. Default is 1. -1 = number of cores -1')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted files')
parser.add_argument('--debug',action='store_true',default=False,help='Run in verbose mode and with one core for debugging')
parser.add_argument('--verbose',action='store_true',default=False,help='Verbose mode')
parser.add_argument('--overwrite',action = 'store_true',default=False,help='Overwrite all files rather than re-using exisiting files. Does not re-extract images. NB! does this by deleting --temp_folder, --out_graphical, --out_contours and all their content')
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
this_script_folder = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if args.categories is None: 
    args.categories = os.path.join(this_script_folder,"utilities","categories.csv")

if not os.path.exists(args.settings_file): 
    cmd = "coco_make_annotation_file.py"
    exit_code = os.system(cmd)
    if exit_code != 0: 
        raise RunTimeError("Command did not finish properly. cmd: "+cmd)

if args.overwrite:
    for folder in [args.temp_folder,args.out_graphical,args.out_contours]:
        oi.delete_folder_with_content(folder)

os.makedirs(args.out_graphical,exist_ok=True)
os.makedirs(args.out_contours,exist_ok=True)
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
segment_settings = oi.excel_to_segment_settings(args.settings_file)

if args.stitch: 
    for image_info in raw_imgs: 
        image_info.get_extracted_files_path(extract_with_imagej=args.stitch)

annotations = []
if os.path.exists(args.annotations): 
    categories = classes.Categories.load_from_file(args.categories)
    
    annotation_files = os.listdir(args.annotations)
    for a in annotation_files: 
        annotations.append(classes.Annotation.load_from_file(os.path.join(args.annotations,a),categories))
    if args.verbose: print("Found "+str(len(annotations))+ " annotation files")

masks = classes.Mask.get_mask_list(args.masks)

def main(image_info,segment_settings,annotations,masks,info):
    '''
    Process one file of the program 

    Params
    image_info       : Image_info               : Information about file to process
    segment_settings : list of Segment_settings : Information about how to process images
    annotations      : list of Annotations      : Annotations of images
    masks            : list of Mask             : Masks to filter image by 
    info             : str                      : Info about the image that is processed to be printed
    '''
    
    global args
    print_str = str(info)+"\traw_img_path: "+str(image_info.raw_path)+"\t"
    
    print(print_str+"Processing")
    images_paths = image_info.get_extracted_files_path(extract_with_imagej=args.stitch)
    z_stacks = oi.img_paths_to_zstack_classes(images_paths,segment_settings)
    
    if len(masks)>0: 
        use_filter_masks = True
    else: 
        use_filter_masks = False 
    
    new_z_stacks = []
    for i in range(len(z_stacks)):
        z_stacks[i] = z_stacks[i].get_max_projection_zstack(segment_settings)
        if use_filter_masks:
            new_z_stacks = new_z_stacks + z_stacks[i].filter_w_mask(masks)
    
    if use_filter_masks: 
        z_stacks = new_z_stacks
    
    for i in range(len(z_stacks)): 
        z_stacks[i].make_masks()
        z_stacks[i].make_combined_masks(use_filter_masks)
        z_stacks[i].find_contours()
        z_stacks[i].group_contours()
        z_stacks[i].add_annotations(annotations)
        z_stacks[i].split_on_annotations()
        z_stacks[i].group_contours()
        if not use_filter_masks: 
            z_stacks[i].is_inside_combined()
        z_stacks[i].check_annotations()
        z_stacks[i].update_contour_stats()
        z_stacks[i].measure_channels()
        z_stacks[i].write_contour_info()
        z_stacks[i].to_pdf()
        if args.verbose: z_stacks[i].print_all()
    
    return None

if args.cores is -1:
    args.cores = mp.cpu_count()-1

tot_images = len(raw_imgs)

out = []
if args.cores==1: 
    for i in range(0,len(raw_imgs)):
        info = str(i+1)+"/"+str(tot_images)
        out.append(main(raw_imgs[i],segment_settings,annotations,masks,info))
else: 
    pool = mp.Pool(args.cores)

    for i in range(0,len(raw_imgs)):
        info = str(i+1)+"/"+str(tot_images)
        out.append(pool.apply_async(main,args=(raw_imgs[i],segment_settings,annotations,masks,info)))

    pool.close()
    pool.join()

    out = [x.get() for x in out]

