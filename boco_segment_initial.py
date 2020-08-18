#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Segment organoids based on positions determined with ORGAI, a program that uses a neural net to find position and classification of organoid pictures')
parser.add_argument('--z_planes',default="rawdata",type=str,help='Folder with all z planes, recursive. Taken with EVOS 2x objective and following naming scheme: {exp}_{plate}_{day}*_0_{well}f00d4.TIF')
parser.add_argument('--session_file',type=str,default='ORGAI_session_file.txt',help='Created session file of randomly ordered wells that links annotations and projections and is used in other ORGAI scripts')
parser.add_argument('--out_projections',type=str,default = 'min_projections', help='Output folder for minimal projections')
parser.add_argument('--out_annotations',type=str,default = 'annotations', help='Output folder for annotations')
parser.add_argument('--out_segmented_well',type=str,default = 'graphic_out_segment_raw', help='Output folder for graphical segmentation')
parser.add_argument('--minimum_size',type = int,default=300,help='The minimum size of organoids. Default: 300.')
parser.add_argument('--categories',type=str,help='File to load category information from. Default is to load it from default file in repository: config/boco_categories.csv')
parser.add_argument('--AI_folder',type=str,default=None,help='Path to folder with AI weights used to predict images. Default is to find in repository: config/AI_train_results')
parser.add_argument('--out_treatment_xlsx',type=str,default='treatment_info.xlsx',help='Excel file where the user can submit treatment information. Only made if does not exist.')
parser.add_argument('--out_process_annotations_rscript',type=str,default='process_annotations.R',help='Copy R script for processing annotations into this location.')
parser.add_argument('--dryrun',action='store_true',default=False,help='Do everything except processing images.')
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores minus 1.')
parser.add_argument('--verbose',action='store_true',default=False,help="Print statements about program state as it runs")
parser.add_argument('--debug',action='store_true',default=False,help='debug mode, no parallel processing and verbose')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted wells')
parser.add_argument('--overwrite',action = 'store_true',help='Re-process images that have annotations.')
args = parser.parse_args()

########################
# Imports 
########################
import time
import datetime
import os
import shutil
import random
import multiprocessing as mp 

import cv2
import numpy as np 
import pandas as pd

from coco_package import image_processing,info,raw_image_read,AI_functions

########################
# setup 
########################
if args.debug: 
    args.verbose = True
    args.cores = 1

if args.verbose: 
    image_processing.set_verbose()
    info.set_verbose()
    raw_image_read.set_verbose()
    AI_functions.set_verbose()
    print("Running boco_segment_initial.py in verbose mode")
    print("opencv version: "+str(cv2.__version__))

##############################
# Run program
##############################
this_script_folder = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if args.categories is None: 
    args.categories = os.path.join(this_script_folder,"config","boco_categories.csv")

os.makedirs(args.out_projections,exist_ok=True)
os.makedirs(args.out_segmented_well,exist_ok=True)
os.makedirs(args.out_annotations,exist_ok=True)

categories = info.Categories.load_from_file(args.categories)

stacks = raw_image_read.Image_evos.find_stacks(args.z_planes)

df = pd.DataFrame(columns = ["id","root_image","file_image","root_annotation","file_annotation","manually_reviewed"])
image_ids = list(set(stacks["id"]))
stacks_per_image = len(stacks.index)/len(image_ids)
n_wells = len(image_ids)

image_ids.sort()

if not args.n_process is None:
    image_ids = image_ids[0:args.n_process]

print("found {n_stacks} stacks belonging to {n_wells} wells, {avg_stacks} stacks per well\n".format(n_stacks = len(stacks.index),n_wells = n_wells,avg_stacks = stacks_per_image))

ai_predict = AI_functions.AI_predict(AI_functions.AI(args.AI_folder))

def main(index,image_numb,tot_images,stacks,categories,ai_predict):
    '''
    Process one image of the program 

    Params
    index      : str              : index of the image to process
    image_numb : int              : number of image that is processed
    tot_images : int              : total images that are processed
    stacks     : pandas.DataFrame : Data frame containing the path to stacks 
    categories : Categories       : Categories relevant for this set of images
    ai_predict : AI_predict       : Instance of object to predict object class with AI
    '''

    min_projection_name = index+".png"
    min_projection_path = os.path.join(args.out_projections,min_projection_name)
    annotation_name = min_projection_name.replace(".png",".txt")
    annotation_path = os.path.join(args.out_annotations,annotation_name)
    
    image_info = pd.Series({"id":index,"root_image":args.out_projections,"file_image":min_projection_name,"root_annotation":args.out_annotations,"file_annotation":annotation_name,"manually_reviewed":"False"})
    print_info = "Processing image: "+str(image_numb)+"/"+str(tot_images)+" id: "+index
    
    if (os.path.isfile(os.path.join(args.out_annotations,annotation_name))):
        if not args.overwrite:
            print(print_info + "\t did nothing, annotation already exist",flush=True)
            return image_info

    print(print_info+ "\t finding annotation centers ",flush=True)
    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d-%H:%M")
    changelog = today+" segmented_images\n"
    
    z_planes = list(stacks["full_path"][stacks['id']==index])
    channel = raw_image_read.Image_evos(z_planes).get_channel(min_projection_path,categories)
    
    if args.verbose: print("\tFinding contours",flush=True)
    channel.find_contours(min_contour_area = args.minimum_size)
    if args.verbose: print("\tSplitting contours",flush=True)
    channel.split_contours()
    if args.verbose: print("\tFinding distance centers",flush=True)
    channel.find_distance_centers(erode = None,halo = None,min_size = None)
    if args.verbose: print("\tClassifying objects",flush=True)
    channel.classify_objects(ai_predict) 
    if args.verbose: print("\tUpdating contour stats",flush=True)
    channel.update_contour_stats()
    if args.verbose: print("\tMeasure channel",flush=True)
    channel.measure_channels([channel])
    if args.verbose: print("\tMaking image with contours",flush=True)
    channel.make_img_with_contours(args.out_segmented_well,auto_max = True,scale_bar = False,colorize = False,add_distance_centers = False,add_contour_numbs = True)
    if args.verbose: print("\tWriting annotation file\n",flush=True)
    channel.write_annotation_file(args.out_annotations,add_to_changelog = "Initial segmentation")
     
    return image_info

if args.cores is None:
    cores = mp.cpu_count()-1
else:
    cores = args.cores

if not args.dryrun: 
    if cores==1: 
        for i in range(0,len(image_ids)):
            image_info = main(image_ids[i],i+1,len(image_ids),stacks,categories,ai_predict)
            df = df.append(image_info,ignore_index=True)
    else: 
        pool = mp.Pool(cores)

        image_info = []
        for i in range(0,len(image_ids)):
            image_info.append(pool.apply_async(main,args=(image_ids[i],i+1,len(image_ids),stacks,categories,ai_predict)))

        pool.close()
        pool.join()

        image_info = [x.get() for x in image_info]

        for i in image_info: 
            df = df.append(i,ignore_index=True)

df = df.sample(frac=1).reset_index(drop=True) #Suffle rows
df.to_csv(args.session_file,index=False)

if not os.path.exists(args.out_treatment_xlsx):
    temp = df['id'].str.split('_d',n=1,expand=True)
    well_id = temp.iloc[:,0]
    well_id = list(set(well_id))
    well_id.sort()
    meta_data_xlsx = pd.DataFrame(columns = ['well_id','treatment'])
    meta_data_xlsx['well_id'] = well_id

    meta_data_xlsx.to_excel(args.out_treatment_xlsx,index=False)

if not os.path.exists(args.out_process_annotations_rscript):
    orgai_folder, this_script_name = os.path.split(os.path.realpath(__file__))
    rscript_current_path = os.path.join(orgai_folder,"config",'process_annotations.R')
    shutil.copyfile(rscript_current_path,args.out_process_annotations_rscript)



