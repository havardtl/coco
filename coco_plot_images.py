#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Extract channels from confocal images, make them into projections and plot in pdf.')
parser.add_argument('--raw_data',default="raw/rawdata",type=str,help='Path to raw image. Can use all formats accepted by your installed version of bftools. ')
parser.add_argument('--extracted_images_folder',type=str,default='raw/extracted_raw',help="Folder with extracted images.")
parser.add_argument('--annotation_file',type=str,default='annotation1.xlsx',help="Excel file with annotation data for files. Make it with coco_make_annotation_file.py.")
parser.add_argument('--projections_pdf_folder',type=str,default = 'graphical/projections_pdf', help='Output folder for pdfs with maximal projections')
parser.add_argument('--projections_raw_folder',type=str,default = 'graphical/projections_raw', help='Output folder for raw maximal projections')
parser.add_argument('--temp_folder',type=str,default='raw/temp',help="temp folder for storing temporary images. Must not exist before startup. default: ./coco_temp")
parser.add_argument('--categories',type=str,help='File to load category information from. Default is to load it from default file in utilities/coco_categories.csv')
parser.add_argument('--channel_colors',type=str,default = "(0,255,0),(255,0,255),(255,0,0),(0,0,255)",help='Colors to use for plotting channels. colors in BGR format. default: (0,255,0),(255,0,255),(255,0,0),(0,0,255)')
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores minus 1.')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted files')
parser.add_argument('--verbose',action='store_true',default=False,help='Verbose mode')
parser.add_argument('--debug',action='store_true',default=False,help='Run in verbose mode and with one core for debugging')
parser.add_argument('--overwrite',action = 'store_true',default=False,help='Overwrite all files rather than re-using exisiting files. NB! does this by deleting --temp_folder, --projections_raw_folder, --projections_pdf_folder and all their content')
parser.add_argument('--max_size',type=int,default=1500*1500,help="Maximal size in total pixels of images in output pdf. NB! Unstable over 5000*5000")
args = parser.parse_args()

########################
# Imports 
########################
import os
import multiprocessing as mp 

import pandas as pd
import cv2
import pickle
import shutil

import utilities.classes as classes

##############################
# Run program
##############################
this_script_folder = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if args.categories is None: 
    args.categories = os.path.join(this_script_folder,"utilities","coco_categories.csv")

if not os.path.exists(args.annotation_file): 
    cmd = "coco_make_annotation_file.py"
    exit_code = os.system(cmd)
    if exit_code != 0: 
        raise RunTimeError("Command did not finish properly. cmd: "+cmd)

if args.overwrite: 
    for folder in [args.temp_folder,args.projections_raw_folder,args.projections_pdf_folder]:
        shutil.rmtree(folder)

if args.debug: 
    args.cores = 1
    args.verbose = True
    
classes.VERBOSE = args.verbose
classes.TEMP_FOLDER = args.temp_folder 

pdf_processed_images_folder = os.path.join(args.temp_folder,"pdf_processed_images")

os.makedirs(args.projections_pdf_folder,exist_ok = True)
os.makedirs(args.projections_raw_folder,exist_ok=True)
os.makedirs(args.temp_folder,exist_ok = True)
os.makedirs(args.extracted_images_folder,exist_ok=True)
os.makedirs(pdf_processed_images_folder,exist_ok=True)

pickle_path = os.path.join(args.temp_folder,"coco_plot_images_df.pickle")

raw_imgs = []
for i in os.listdir(args.raw_data): 
    raw_path = os.path.join(args.raw_data,i)
    img_i = classes.Image_czi(raw_path,args.temp_folder,args.extracted_images_folder)
    raw_imgs.append(img_i)
raw_imgs.sort()

print("Found {n_files} to process".format(n_files = len(raw_imgs)),flush=True)

categories = classes.Categories.load_from_file(args.categories)

if not args.n_process is None:
    raw_imgs = raw_imgs[0:args.n_process]

segment_settings = classes.Segment_settings.excel_to_segment_settings(args.annotation_file)

def main(image_info,info,segment_settings,categories):
    '''
    Process one file of the program 

    Params
    image_info       : Image_info               : Information about file to process
    info             : str                      : Info about the image that is processed to be printed
    segment_settings : list of Segment_settings : Information about how to process images
    '''
    global args

    print(info+"\traw_img_path: "+image_info.raw_path+"\t Making projections",flush=True)
    
    images_paths = image_info.get_extracted_files_path(extract_method="aicspylibczi",max_projection = True)

    if args.verbose: print("Getting info about z_stacks: ",flush=True)
    df = pd.DataFrame()
    
    z_stacks = image_info.get_z_stack(segment_settings,categories,extract_method="aicspylibczi",max_projection = True)
    
    for z in z_stacks:
        if args.verbose: 
            z.print_all()
        z.make_projections(save_folder = args.projections_raw_folder)
        df = pd.concat([df,z.get_projections_data()],ignore_index=True)
    if args.verbose: print(df,flush=True)
    
    return df

if os.path.exists(pickle_path): 
    with open(pickle_path,'rb') as f: 
        df = pickle.load(f)
    print("Loading info from previously made pickle object: "+pickle_path,flush=True)
else: 
    if args.cores is None:
        cores = mp.cpu_count()-1
    else:
        cores = args.cores

    tot_images = len(raw_imgs)

    image_info = []
    if cores==1: 
        for i in range(0,len(raw_imgs)):
            info = str(i+1)+"/"+str(tot_images)
            image_info.append(main(raw_imgs[i],info,segment_settings,categories))
    else: 
        pool = mp.Pool(cores)

        for i in range(0,len(raw_imgs)):
            info = str(i+1)+"/"+str(tot_images)
            image_info.append(pool.apply_async(main,args=(raw_imgs[i],info,segment_settings,categories)))

        pool.close()
        pool.join()

        image_info = [x.get() for x in image_info]

    df = pd.DataFrame()
    for i in image_info: 
        df = pd.concat([df,i],ignore_index=True)
        
    with open(pickle_path,"wb") as f: 
        pickle.dump(df,f)
    print("Wrote info about extracted images to pickle: "+pickle_path,flush=True)

print(type(df),flush=True)
print(df["img_dim"].mode(),flush=True)

most_common_img_dim = str(df["img_dim"].mode()[0])
image_dim_goal = classes.Pdf.img_dim_str_to_tuple(most_common_img_dim)
image_dim_goal = (int(image_dim_goal[0]),int(image_dim_goal[1]))

print("Adding annotation info from: "+args.annotation_file,flush=True)
annotation = pd.read_excel(args.annotation_file,sheet_name = classes.ANNOTATION_SHEET)
print(annotation,flush=True)
annotation.drop("full_path",axis="columns",inplace=True)
df = df.merge(annotation,on="file_id",how="left")

plot_vars = pd.read_excel(args.annotation_file,sheet_name = classes.PLOT_VARS_SHEET)

df["pdf_file"] = os.path.splitext(os.path.split(args.annotation_file)[1])[0]

print("Making pdf from this info:",flush=True)
print(df)

file_vars = ["pdf_file"] + list(plot_vars.loc[plot_vars["plot_axis"]=="file","variable"]) 
x_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="x","variable"]) 
y_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="y","variable"]) 
image_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="image","variable"])

file_vars_sortdirs = [True]
x_vars_sortdirs = list(plot_vars.loc[plot_vars["plot_axis"]=="x","sort_ascending"])
y_vars_sortdirs = list(plot_vars.loc[plot_vars["plot_axis"]=="y","sort_ascending"])

sort_directions = file_vars_sortdirs+y_vars_sortdirs+x_vars_sortdirs

classe.Pdf.plot_images_pdf(args.projections_pdf_folder,df,file_vars,x_vars,y_vars,image_vars,image_dim_goal,pdf_processed_images_folder,sort_directions)

