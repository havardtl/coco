#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Process one of the confocal images with a multitude of segmentation settings to find the best settings. ')
parser.add_argument('--raw_data',default="raw/rawdata",type=str,help='Path to raw image or folder with raw images. If a folder is submitted the alfabetically first is choosen. Can use all formats accepted by your installed version of bftools. ')
parser.add_argument('--extracted_images_folder',type=str,default='raw/extracted_raw',help="Folder with extracted images.")
parser.add_argument('--out_folder',type=str,default = 'graphical/find_segment_settings', help='Output folder for segmentation test results')
parser.add_argument('--annotation_file',type=str,default='annotation1.xlsx',help="Excel file with test settings to try all combinations of. Make it with coco_make_annotation_file.py.")
parser.add_argument('--z_toshow',type = str,default="i3",help='z-slices to include. "1:3" = [1,2,3],"1,5,8" = [1,5,8] and "i3" = 3 evenly choosen from range. Default = "i3"')
parser.add_argument('--temp_folder',type=str,default='raw/temp',help="temp folder for storing temporary images. default: ./coco_temp")
parser.add_argument('--max_size',type=int,default=1000*1000,help="Maximal size for images in PDF in MP")
parser.add_argument('--debug',action='store_true',default=False,help='Run in verbose mode')
parser.add_argument('--verbose',action='store_true',default=False,help='Verbose mode')
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

##############################
# Run program
##############################
if not os.path.exists(args.annotation_file): 
    cmd = "coco_make_annotation_file.py"
    exit_code = os.system(cmd)
    if exit_code != 0: 
        raise RunTimeError("Command did not finish properly. cmd: "+cmd)

pdf_save_folder = os.path.join(args.out_folder,"segmented_pdfs")
viewing_automax_false = os.path.join(args.temp_folder,"find_settings_automax_false")
viewing_automax_true  = os.path.join(args.temp_folder,"find_settings_automax_true")
mask_save_folder = os.path.join(args.temp_folder,"test_settings_masks")
pdf_processed_images_folder = os.path.join(args.temp_folder,"pdf_processed_images")

oi.delete_folder_with_content(viewing_automax_false)
oi.delete_folder_with_content(viewing_automax_true)
oi.delete_folder_with_content(mask_save_folder)
oi.delete_folder_with_content(pdf_processed_images_folder)

os.makedirs(args.out_folder,exist_ok=True)
os.makedirs(args.temp_folder,exist_ok=True)
os.makedirs(args.extracted_images_folder,exist_ok=True)
os.makedirs(pdf_save_folder,exist_ok=True)
os.makedirs(mask_save_folder,exist_ok=True)
os.makedirs(viewing_automax_false,exist_ok=True)
os.makedirs(viewing_automax_true,exist_ok=True)
os.makedirs(pdf_processed_images_folder,exist_ok=True)

if args.debug: 
    args.verbose = True

classes.VERBOSE = args.verbose
classes.TEMP_FOLDER = args.temp_folder

if os.path.isdir(args.raw_data): 
    raw_imgs = os.listdir(args.raw_data)
    print("Found {n} files in raw folder".format(n = len(raw_imgs)))

    raw_imgs.sort()
    raw_img_path = os.path.join(args.raw_data,raw_imgs[0])
elif os.path.isfile(args.raw_data):
    raw_img_path = args.raw_data
else: 
    raise ValueError("--raw_data did not supply a valid path or folder")

print("Processing image: "+raw_img_path)

img_id = os.path.splitext(os.path.split(raw_img_path)[1])[0]

extracted_images_folder_this_img = os.path.join(args.extracted_images_folder,img_id)

images_paths_file = "files_info.txt"
if os.path.isfile(os.path.join(extracted_images_folder_this_img,images_paths_file)): 
    print("Images already extracted from raw files, using those.")
    with open(os.path.join(extracted_images_folder_this_img,images_paths_file),'r') as f: 
        images_paths = f.read().splitlines()
else:
    print("Extracting images...")
    images_paths = oi.get_images_bfconvert(raw_img_path,extracted_images_folder_this_img,images_paths_file,args.verbose)

if args.verbose: print("loading segment settings from "+args.annotation_file)
segment_settings = oi.excel_to_segment_settings(args.annotation_file)

if args.verbose: print("Converting image paths to channels.",end = " ")
channels = oi.img_paths_to_channel_classes(images_paths,segment_settings)

if args.verbose: print("Making test settings from file: "+str(args.annotation_file))
order = ["contrast","auto_max","open_kernel","close_kernel","thresh_type","thresh_upper","thresh_lower"]
test_settings = oi.make_test_settings(args.annotation_file,order)
print("Made {n} test settings".format(n=len(test_settings)))

df = pd.DataFrame({"channel":channels,"z_toshow":False,"z_index":None})

all_z_indexes = []
for i in channels: 
    all_z_indexes.append(int(i.z_index))
all_z_indexes = list(set(all_z_indexes))
all_z_indexes.sort()

choosen_z = oi.choose_z_slices(all_z_indexes,args.z_toshow)

for i in df.index:
    df.loc[i,"z_index"] = int(df.loc[i,"channel"].z_index) 
    df.loc[i,"channel_index"] = df.loc[i,"channel"].channel_index
    if df.loc[i,"z_index"] in choosen_z:
        df.loc[i,"z_toshow"] = True
df["channel_index"] = df["channel_index"].astype(int)

print("Out of {tot_z} z_slices we are displaying {n}".format(tot_z = len(df.index),n = sum(df["z_toshow"])))

df = df.loc[df["z_toshow"],]
print(df)

for i in df.index: 
    df.loc[i,"img_dim"] = str(df.loc[i,"channel"].get_image().shape)

img_dim = (None,None)
temp = str(df["img_dim"].mode()[0]).replace("(","").replace(")","").split(",",1)
img_dim = (int(temp[0]),int(temp[1]))

setting_counter = 0

for i in df["channel_index"].unique():
    this_channel = df.loc[df["channel_index"]==i,]

    pdf_save_path = os.path.join(pdf_save_folder,img_id+"_C"+str(i)+".pdf")
    print("Making PDF: "+pdf_save_path) 

    pdf_imgs = []
    x = 0
    y = 0
    x_vars = ["z_index"]
    y_vars = ["contrast","auto_max","thresh_type","thresh_upper","thresh_lower","open_kernel","close_kernel"]
    
    image_vars = ["file_id"]
    data = {}
    for field in x_vars+y_vars+image_vars:
        data[field] = None

    for j in this_channel.index:
        auto_max = False 
        c = this_channel.loc[j,"channel"]
        img = c.get_img_for_viewing(scale_bar=False,auto_max=auto_max,colorize=True)
        img_path = os.path.join(viewing_automax_false,c.file_id+".png")
        cv2.imwrite(img_path,img)

        data["file_id"]  = img_path 
        data["z_index"]  = this_channel.loc[j,"z_index"]
        data["auto_max"] = auto_max 
        pdf_imgs.append(classes.Image_in_pdf(x,y,img_path,data.copy(),x_vars,y_vars,image_vars,pdf_processed_images_folder,img_dim,args.max_size))
        x = x + 1 
    y = y +1
    x = 0

    for j in this_channel.index:
        auto_max = True
        c = this_channel.loc[j,"channel"]
        img = c.get_img_for_viewing(scale_bar=False,auto_max=auto_max,colorize=True)
        img_path = os.path.join(viewing_automax_true,c.file_id+".png")
        cv2.imwrite(img_path,img)

        data["file_id"]  = img_path  
        data["z_index"]  = this_channel.loc[j,"z_index"]
        data["auto_max"] = auto_max
        pdf_imgs.append(classes.Image_in_pdf(x,y,img_path,data.copy(),x_vars,y_vars,image_vars,pdf_processed_images_folder,img_dim,args.max_size))
        x = x + 1 
    y = y + 1 
    x = 0
    
    for s in test_settings: 
        for j in this_channel.index:
            c = this_channel.loc[j,"channel"]
            
            c.make_mask(s)
            mask = c.get_mask()
            mask_path = os.path.join(mask_save_folder,c.file_id+"_"+str(setting_counter)+".png")
            setting_counter = setting_counter + 1
            cv2.imwrite(mask_path,mask)
            data["file_id"]  = mask_path
            data["z_index"]  = this_channel.loc[j,"z_index"]
            data.update(s.get_dict())
            pdf_imgs.append(classes.Image_in_pdf(x,y,mask_path,data.copy(),x_vars,y_vars,image_vars,pdf_processed_images_folder,img_dim,args.max_size))
            x = x + 1 
        y = y + 1
        x = 0

    pdf = classes.Pdf(pdf_save_path,pdf_imgs)
    pdf.make_pdf()



