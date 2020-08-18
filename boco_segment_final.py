#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Resegment organoids after manual change of annotaiton to update organoid stats')
parser.add_argument('--z_planes',default = 'rawdata',type=str,help='Folder with all z planes, recursive. Taken with EVOS 2x objective and following naming scheme: {exp}_{plate}_{day}*_0_{well}f00d4.TIF')
parser.add_argument('--session_file',type=str,default='ORGAI_session_file.txt',help='Session file made with ORGAI_init.py that contains paths to all annotation files')
parser.add_argument('--annotations',type=str,default='annotations',help="Folder with annotation of channel images. NB! Annotations not manually reviewed are not added.")
parser.add_argument('--out_folder',type=str,default = 'segmented_post_manual', help='Out put folder for single organoid images and info file')
parser.add_argument('--categories',type=str,help='File to load category information from. Default is to load it from default file in config/boco_categories.csv')
parser.add_argument('--single_objects',type = str, default = None,help='Create single object images in this folder for training AI, default is to not make single object images.')
parser.add_argument('--debug',action='store_true',default=False,help='debug mode, no parallel processing and verbose')
parser.add_argument('--verbose',action='store_true',default=False,help="Print statements about program state as it runs")
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores in machine minus 1.')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted wells')
args = parser.parse_args()

########################
# Imports 
########################
import os
import datetime
import multiprocessing as mp 

import numpy as np 
import pandas as pd
import cv2

from coco_package import image_processing
from coco_package import raw_image_read
from coco_package import info

########################
# setup 
########################
this_script_folder = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if args.categories is None: 
    args.categories = os.path.join(this_script_folder,"config","boco_categories.csv")

if args.debug: 
    args.verbose = True
    args.cores = 1

if args.verbose: 
    image_processing.set_verbose()
    raw_image_read.set_verbose()
    print("Running boco_segment_final.py in verbose mode",flush=True)
    
if args.single_objects is not None:
    if os.path.exists(args.single_objects):
        print("WARNING: Single objects folder '"+args.single_objects+"' already exists.")
        #raise ValueError("Single objects folder '"+args.single_objects"' already exists.")
    os.makedirs(args.single_objects,exist_ok=True)
    
##############################
# Run program
##############################

annotations_folder = os.path.join(args.out_folder,'annotations')
os.makedirs(annotations_folder,exist_ok=True)
graphic_segmentation_path = os.path.join(args.out_folder,'graphic_segmentation')
os.makedirs(graphic_segmentation_path,exist_ok=True)

stacks = raw_image_read.Image_evos.find_stacks(args.z_planes)

session = info.Session.from_file(args.session_file)
session.find_missing(stacks["id"])

if not args.n_process is None:
    session.sort()
    session.select_first(args.n_process)
    if args.verbose: print("Only selecting "+str(args.n_process)+" images, as set with --n_process",flush=True)

print("Found "+str(session.get_n())+" images. Of which "+str(session.get_n_reviewed())+" is already reviewed",flush=True)

annotations = []
if os.path.exists(args.annotations): 
    categories = info.Categories.load_from_file(args.categories)
    
    annotation_files = os.listdir(args.annotations)
    for a in annotation_files: 
        annotations.append(info.Annotation.load_from_file(os.path.join(args.annotations,a),categories))
    if args.verbose: print("Found "+str(len(annotations))+ " annotation files\n")

def main(image_info,process_info,stacks,annotations,categories,annotations_folder):
    '''
    Process one EVOS z_stack
    
    Params
    image_info          : Image_info          : information about image to process
    process_info        : str                 : info about image that is currently being processed
    stacks              : pd.DataFrame        : Dataframe with raw paths to all images
    annotations         : list of Annotation  : list of class with annotational data
    categories          : Categories          : class with information about categories
    annotations_folder  : str                 : Path to where annotations are written
    '''
    if args.verbose: print("",flush=True)
    print(process_info+ "\tProcessing image",flush=True)
   
    z_planes = list(stacks.loc[stacks['id']==image_info.id,"full_path"])
    
    channel = raw_image_read.Image_evos(z_planes).get_channel(image_info.img_path,categories)
    if args.verbose: print("\tFinding contours",flush=True)
    channel.find_contours()
    if args.verbose: print("\tAdding annotations")
    channel.add_annotation(annotations,match_file_id=True)
    if args.verbose: print("\tSplitting on annotations",flush=True)
    channel.split_on_annotations()
    if args.verbose: print("\tChecking annotations",flush=True)
    channel.check_annotation()
    if args.verbose: print("\tUpdating stats",flush=True)
    channel.update_contour_stats()
    if args.verbose: print("\tMeasure channel",flush=True)
    channel.measure_channels([channel])
    if args.verbose: print("\tMaking graphical segmentation",flush=True)
    channel.make_img_with_contours(graphic_segmentation_path,auto_max = True,scale_bar = False,colorize = False,add_distance_centers = False,add_contour_numbs = False)
    if args.single_objects is not None: 
        if args.verbose: print("\tMaking single objects images")
        channel.write_single_objects(args.single_objects,merge_categories={"Junk":"None"})
    if args.verbose: print("\tWriting annotation file",flush=True)
    channel.write_annotation_file(annotations_folder,add_to_changelog = "Segmented_post_manual")
   
if args.cores is None:
    cores = mp.cpu_count()-1
else:
    cores = args.cores

session.reset_index()
if cores==1: 
    while True:
        main(session.get_img_info(),session.get_process_info(),stacks,annotations,categories,annotations_folder)
        if session.next_index():
            break
        
else: 
    pool = mp.Pool(cores)
    
    result = []
    while True:
        result.append(pool.apply_async(main,args=(session.get_img_info(),session.get_process_info(),stacks,annotations,categories,annotations_folder)))
        if session.next_index():
            break
            
    pool.close()
    pool.join()
    
    result = [x.get() for x in result]


