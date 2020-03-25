#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Resegment organoids after manual change of annotaiton to update organoid stats')
parser.add_argument('--z_planes',default = 'rawdata',type=str,help='Folder with all z planes, recursive. Taken with EVOS 2x objective and following naming scheme: {exp}_{plate}_{day}*_0_{well}f00d4.TIF')
parser.add_argument('--session_file',type=str,default='ORGAI_session_file.txt',help='Session file made with ORGAI_init.py that contains paths to all annotation files')
parser.add_argument('--out_folder',type=str,default = 'segmented_post_manual', help='Out put folder for single organoid images and info file')
parser.add_argument('--create_single_organoids',action = 'store_true',default=False,help='Create single organoid images, default is false')
parser.add_argument('--debug',action='store_true',default=False,help='debug mode, no parallel processing and verbose')
parser.add_argument('--verbose',action='store_true',default=False,help="Print statements about program state as it runs")
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores in machine minus 1.')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted wells')
parser.add_argument('--categories',type=str,help='File to load category information from. Default is to load it from default file in utilities/boco_categories.csv')
parser.add_argument('--annotations',type=str,default='annotations',help="Folder with annotation of channel images")
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

import utilities.image_processing_functions as oi
import utilities.classes as classes

########################
# setup 
########################
this_script_folder = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if args.categories is None: 
    args.categories = os.path.join(this_script_folder,"utilities","boco_categories.csv")

if args.debug: 
    args.verbose = True
    args.cores = 1

if args.verbose: 
    oi.VERBOSE = True
    classes.VERBOSE = True
    print("Running boco_segment_final.py in verbose mode")
    
##############################
# Run program
##############################

annotations_folder = os.path.join(args.out_folder,'annotations')
os.makedirs(annotations_folder,exist_ok=True)
graphic_segmentation_path = os.path.join(args.out_folder,'graphic_segmentation')
os.makedirs(graphic_segmentation_path,exist_ok=True)

stacks = oi.find_stacks(args.z_planes)

df, index = oi.load_session_file(args.session_file)
df["image_numb"] = list(range(1,len(df.index)+1))

if not args.n_process is None:
    df.sort_index(axis="index",inplace = True)
    df = df.iloc[0:args.n_process,]
    if args.verbose: print("Only selecting "+str(args.n_process)+" images, as set with --n_process")

print("Found "+str(len(df.index))+" images. Of which "+str(df["manually_reviewed"].sum())+" is already reviewed")

annotations = []
if os.path.exists(args.annotations): 
    categories = classes.Categories.load_from_file(args.categories)
    
    annotation_files = os.listdir(args.annotations)
    for a in annotation_files: 
        annotations.append(classes.Annotation.load_from_file(os.path.join(args.annotations,a),categories))
    if args.verbose: print("Found "+str(len(annotations))+ " annotation files\n")

def main(df,index,stacks,annotations,categories,annotations_folder):
    if args.verbose: print("")
    print(str(df.loc[index,'image_numb'])+"/"+str(len(df.index))+"\tid: "+index + "\tProcessing image")
   
    min_projection_name = df.loc[index,"file_image"]
    min_projection_path = os.path.join(df.loc[index,"root_image"],min_projection_name)
    
    z_planes = list(stacks["full_path"][stacks['id']==index])
    edges = oi.find_edges(z_planes)
    
    channel = classes.Channel(min_projection_path,channel_index = 0,z_index = 0,color = (255,255,255),categories = categories)
    channel.mask = edges
    if args.verbose: print("\tFinding contours")
    channel.find_contours()
    if args.verbose: print("\tAdding annotations")
    channel.add_annotation(annotations,match_file_id=True)
    if args.verbose: print("\tSplitting on annotations")
    channel.split_on_annotations()
    if args.verbose: print("\tChecking annotations")
    channel.check_annotation()
    if args.verbose: print("\tUpdating stats")
    channel.update_contour_stats()
    if args.verbose: print("\tMeasure channel")
    channel.measure_channels([channel])
    if args.verbose: print("\tMaking graphical segmentation")
    channel.make_img_with_contours(graphic_segmentation_path,auto_max = True,scale_bar = False,colorize = False,add_distance_centers = False,add_contour_numbs = False)
    if args.verbose: print("\tWriting annotation file")
    channel.write_annotation_file(annotations_folder,add_to_changelog = "Segmented_post_manual")
   
if args.cores is None:
    cores = mp.cpu_count()-1
else:
    cores = args.cores
    
if cores==1: 
    for index in df.index:
        main(df,index,stacks,annotations,categories,annotations_folder)
        
else: 
    pool = mp.Pool(cores)
    
    result = []
    for index in df.index:
        result.append(pool.apply_async(main,args=(df,index,stacks,annotations,categories,annotations_folder)))

    pool.close()
    pool.join()
    
    result = [x.get() for x in result]


