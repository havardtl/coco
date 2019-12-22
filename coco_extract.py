#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Extract images from czi files to .ome.tif.')
parser.add_argument('--raw_folder',default="raw/rawdata",type=str,help='Folder with all raw data. Does not look for files in sub-folders. Can use all formats accepted by your installed version of bftools.')
parser.add_argument('--temp_folder',type=str,default='raw/temp',help="temp folder for storing temporary files.")
parser.add_argument('--extracted_images_folder',type=str,default='raw/extracted_raw',help="Folder with extracted images.")
parser.add_argument('--stitch',default=False,action="store_true",help='Switch that turns on stitching of images using ImageJ. NB! Very experimental')
parser.add_argument('--cores',type=int,default=1,help='Number of cores to use. Default is 1. -1 = number of cores -1')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted files')
parser.add_argument('--debug',action='store_true',default=False,help='Run in verbose mode and with one core for debugging')
parser.add_argument('--verbose',action='store_true',default=False,help='Verbose mode')
args = parser.parse_args()

########################
# Imports 
########################
import os
import multiprocessing as mp 

import utilities.image_processing_functions as oi
import utilities.classes as classes

########################
# Run program
########################

os.makedirs(args.temp_folder,exist_ok = True)
os.makedirs(args.extracted_images_folder,exist_ok = True)

pickle_folder = os.path.join(args.temp_folder,"segment_pickles")
classes.VERBOSE = args.verbose
classes.TEMP_FOLDER = args.temp_folder 

raw_imgs = []
for i in os.listdir(args.raw_folder): 
    raw_path = os.path.join(args.raw_folder,i)
    img_i = classes.Image_info(raw_path,args.temp_folder,args.extracted_images_folder,pickle_folder)
    raw_imgs.append(img_i)
raw_imgs.sort()


def main(image_info,info): 
    '''
    Process one file of the program 

    Params
    image_info       : Image_info               : Information about file to process
    segment_settings : list of Segment_settings : Information about how to process images
    info             : str                      : Info about the image that is processed to be printed
    '''
    print(info +"\t"+image_info.raw_path)
    image_info.get_extracted_files_path(extract_with_imagej=args.stitch)

tot_images = len(raw_imgs)

out_info = []
if args.cores==1: 
    for i in range(0,len(raw_imgs)):
        info = str(i+1)+"/"+str(tot_images)
        out_info.append(main(raw_imgs[i],info))
else: 
    pool = mp.Pool(args.cores)

    for i in range(0,len(raw_imgs)):
        info = str(i+1)+"/"+str(tot_images)
        out_info.append(pool.apply_async(main,args=(raw_imgs[i],info)))

    pool.close()
    pool.join()

    out_info = [x.get() for x in out_info]

