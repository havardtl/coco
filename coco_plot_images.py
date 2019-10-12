#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Extract channels from confocal images, make them into projections and plot in pdf.')
parser.add_argument('--raw_data',default="rawdata",type=str,help='Path to raw image. Can use all formats accepted by your installed version of bftools. ')
parser.add_argument('--annotation_file',type=str,default='annotation1.xlsx',help="Excel file with annotation data for files. Make it with coco_make_annotation_file.py.")
parser.add_argument('--projections_pdf_folder',type=str,default = 'projections_pdf', help='Output folder for pdfs with maximal projections')
parser.add_argument('--projections_raw_folder',type=str,default = 'projections_raw', help='Output folder for raw maximal projections')
parser.add_argument('--temp_folder',type=str,default='coco_temp',help="temp folder for storing temporary images. Must not exist before startup. default: ./coco_temp")
parser.add_argument('--channel_colors',type=str,default = "(0,255,0),(255,0,255),(255,0,0),(0,0,255)",help='Colors to use for plotting channels. colors in BGR format. default: (0,255,0),(255,0,255),(255,0,0),(0,0,255)')
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores minus 1.')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted files')
parser.add_argument('--verbose',action='store_true',default=False,help='Verbose mode')
parser.add_argument('--debug',action='store_true',default=False,help='Run in verbose mode and with one core for debugging')
parser.add_argument('--overwrite',action = 'store_true',default=False,help='Overwrite all files rather than re-using exisiting files. NB! does this by deleting --temp_folder, --projections_raw_folder, --projections_pdf_folder and all their content')
args = parser.parse_args()

########################
# Imports 
########################
import os
import multiprocessing as mp 

import pandas as pd
import cv2
import pickle

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

if args.overwrite: 
    for folder in [args.temp_folder,args.projections_raw_folder,args.projections_pdf_folder]:
        oi.delete_folder_with_content(folder)
    
if args.debug: 
    args.cores = 1
    args.verbose = True

classes.VERBOSE = args.verbose
classes.TEMP_FOLDER = args.temp_folder 
classes.PROJECTIONS_RAW_FOLDER = args.projections_raw_folder

os.makedirs(args.projections_pdf_folder,exist_ok = True)
os.makedirs(args.projections_raw_folder,exist_ok=True)
os.makedirs(args.temp_folder,exist_ok = True)

pickle_path = os.path.join(args.temp_folder,"coco_plot_images_df.pickle")

image_ids_all = os.listdir(args.raw_data)
for i in range(len(image_ids_all)): 
    image_ids_all[i] = os.path.join(args.raw_data,image_ids_all[i])
image_ids_all.sort()

if not args.n_process is None:
    image_ids_all = image_ids_all[0:args.n_process]

print("Found {n_files} to process".format(n_files = len(image_ids_all)))

segment_settings = oi.excel_to_segment_settings(args.annotation_file)

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
    
    print_str = info+"\traw_img_path: "+raw_img_path+"\t"
   
    images_paths_file = "files_info.txt"
    bfconvert_info_str = "_INFO_%s_%t_%z_%c"
    file_ending = ".ome.tiff"

    if os.path.isfile(os.path.join(extracted_images_folder,images_paths_file)): 
        print(print_str+"Images already extracted from raw files, using those to make projections.")
        with open(os.path.join(extracted_images_folder,images_paths_file),'r') as f: 
            images_paths = f.read().splitlines()
    else:
        print(print_str+"Extracting images")
        images_paths = oi.get_images(raw_img_path,extracted_images_folder,images_paths_file,bfconvert_info_str,file_ending)
        
    if args.verbose: print("Getting info about z_stacks: ")
    df = pd.DataFrame()
    z_stacks = oi.img_paths_to_zstack_classes(images_paths,file_ending,segment_settings)
    for z in z_stacks:
        if args.verbose: 
            z.print_all()
        z.make_projections()
        df = pd.concat([df,z.get_projections_data()],ignore_index=True)
    if args.verbose: print(df)
    
    return df

if os.path.exists(pickle_path): 
    with open(pickle_path,'rb') as f: 
        df = pickle.load(f)
    print("Loading info from previously made pickle object: "+pickle_path)
else: 
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

    df = pd.DataFrame()
    for i in image_info: 
        df = pd.concat([df,i],ignore_index=True)
    
    with open(pickle_path,"wb") as f: 
        pickle.dump(df,f)

most_common_img_dim = str(df["img_dim"].mode()[0])
img_dim_goal = most_common_img_dim.replace("(","").replace(")","").split(",",1)
image_dim_goal = (int(img_dim_goal[0]),int(img_dim_goal[1]))
goal_ratio = float(img_dim_goal[1])/float(img_dim_goal[0])

cropped_projection_folder = os.path.join(args.temp_folder,"cropped_projections")
os.makedirs(cropped_projection_folder,exist_ok=True)

for i in df.index:
    img_dim = df.loc[i,"img_dim"]
    if img_dim != most_common_img_dim:
        img_dim = img_dim.split("x")
        img_dim_ratio = float(img_dim[1])/float(img_dim[0])
        
        if img_dim_ratio > goal_ratio:
            new_width = int((img_dim_ratio/goal_ratio)*float(img_dim[0]))
            new_height = int(img_dim[1])
        else: 
            new_width = int(img_dim[0])
            new_height = int((goal_ratio/img_dim_ratio)*float(img_dim[1]))

        new_projection = cv2.imread(df.loc[i,"full_path"])
        new_projection = oi.imcrop2(new_projection,[0,0,new_width,new_height],value=(150,150,150))
        cropped_projection_path = os.path.join(cropped_projection_folder,os.path.split(df.loc[i,"full_path"])[1])
        cv2.imwrite(cropped_projection_path,new_projection)
        df.loc[i,"full_path"] = cropped_projection_path

print("Adding annotation info from: "+args.annotation_file)
annotation = pd.read_excel(args.annotation_file,sheet_name = classes.ANNOTATION_SHEET)
print(annotation)
annotation.drop("file_path",axis="columns",inplace=True)
df = df.merge(annotation,on="file_id",how="left")

plot_vars = pd.read_excel(args.annotation_file,sheet_name = classes.PLOT_VARS_SHEET)

df["pdf_file"] = os.path.splitext(os.path.split(args.annotation_file)[1])[0]

print("Making pdf from this info:")
print(df)

file_vars = ["pdf_file"] + list(plot_vars.loc[plot_vars["plot_axis"]=="file","variable"]) 
x_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="x","variable"]) 
y_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="y","variable"]) 
image_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="image","variable"])

file_vars_sortdirs = [True]
x_vars_sortdirs = list(plot_vars.loc[plot_vars["plot_axis"]=="x","sort_ascending"])
y_vars_sortdirs = list(plot_vars.loc[plot_vars["plot_axis"]=="y","sort_ascending"])

sort_directions = file_vars_sortdirs+y_vars_sortdirs+x_vars_sortdirs

oi.plot_images_pdf(args.projections_pdf_folder,df,file_vars,x_vars,y_vars,image_vars,image_dim = image_dim_goal,sort_ascending = sort_directions)

