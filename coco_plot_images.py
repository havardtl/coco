#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Extract channels from confocal images and plot in pdf.')
parser.add_argument('--raw_data',default="rawdata",type=str,help='Path to raw image. Can use all formats accepted by your installed version of bftools. ')
parser.add_argument('--projections_pdf_folder',type=str,default = 'projections_pdf', help='Output folder for pdfs with maximal projections')
parser.add_argument('--projections_raw_folder',type=str,default = 'projections_raw', help='Output folder for raw maximal projections')
parser.add_argument('--annotation_file',type=str,default='annotation1.xlsx',help="Excel file with annotation data for files. Make it with coco_make_annotation_file.py.")
parser.add_argument('--temp_folder',type=str,default='coco_temp',help="temp folder for storing temporary images. Must not exist before startup. default: ./coco_temp")
parser.add_argument('--channel_colors',type=str,default = "(0,255,0),(255,0,255),(255,0,0),(0,0,255)",help='Colors to use for plotting channels. colors in BGR format. default: (0,255,0),(255,0,255),(255,0,0),(0,0,255)')
parser.add_argument('--cores',type=int,help='Number of cores to use. Default is number of cores minus 1.')
parser.add_argument('--n_process',type=int,help='Process the n first alphabetically sorted files')
args = parser.parse_args()

########################
# Imports 
########################
import os

#if os.path.isdir(args.temp_folder):
#    raise ValueError('--temp_folder already exists. This one is deleted by the program and must be clean')

import pandas as pd
import cv2 

import utilities.general_functions as oc
import utilities.image_processing_functions as oi

##############################
# Run program
##############################
os.makedirs(args.projections_pdf_folder,exist_ok = True)
os.makedirs(args.temp_folder,exist_ok = True)

extracted_images_temp_folder = os.path.join(args.temp_folder,"extracted_raw")
os.makedirs(extracted_images_temp_folder,exist_ok = True)

channel_colors = []
for c in args.channel_colors.split("),("):
    c = c.replace("(","").replace(")","")
    b,g,r = c.split(",",3)
    channel_colors.append((int(b),int(g),int(r)))

df_all = pd.DataFrame()

image_ids_all = []
for root, dirs,files in os.walk(args.raw_data):
    for f in files: 
        image_ids_all.append(os.path.join(root,f))

if not args.n_process is None:
    image_ids_all.sort()
    image_ids_all = image_ids_all[0:args.n_process]

if not os.path.isdir(args.projections_raw_folder): 
    print("Found {n_files} files to process".format(n_files = len(image_ids_all)))
    os.makedirs(args.projections_raw_folder,exist_ok = True)
    oi.make_many_projections(image_ids_all,extracted_images_temp_folder,args.projections_raw_folder,channel_colors,args.cores)
else:
    print("Raw projection folder already exists. Not recreating them.")

#TODO: move everything below here into a function called prepare_pdf_plotting(args.projections_raw_folder,args.projections_pdf_folder,args.temp_folder,args.annotation_file)

projections_path = []
for root, dirs,files in os.walk(args.projections_raw_folder):
    for f in files: 
        projections_path.append(os.path.join(root,f))

projections = pd.DataFrame(data = {"file_path":projections_path})

file_id = []
for i in projections.index:
    file_id.append(os.path.splitext(os.path.split(projections.loc[i,"file_path"])[1])[0])
projections["file_id"] = file_id

projections["file_id"],projections["info"] = projections["file_id"].str.split("_INFO_",1).str
projections["img_dim"],projections["series"],projections["time"],projections["channel"],projections["leveled"] = projections["info"].str.split("_",4).str

most_common_img_dim = projections["img_dim"].value_counts().idxmax()
img_dim_goal = most_common_img_dim.split("x")
image_dim_goal = (int(img_dim_goal[0]),int(img_dim_goal[1]))
goal_ratio = float(img_dim_goal[1])/float(img_dim_goal[0])

cropped_projection_folder = os.path.join(args.temp_folder,"cropped_projections") 
os.makedirs(cropped_projection_folder,exist_ok=True)

for i in projections.index:
    img_dim = projections.loc[i,"img_dim"]
    if img_dim != most_common_img_dim:
        img_dim = img_dim.split("x")
        img_dim_ratio = float(img_dim[1])/float(img_dim[0])
        
        if img_dim_ratio > goal_ratio:
            new_width = int((img_dim_ratio/goal_ratio)*float(img_dim[0]))
            new_height = int(img_dim[1])
        else: 
            new_width = int(img_dim[0])
            new_height = int((goal_ratio/img_dim_ratio)*float(img_dim[1]))

        new_projection = cv2.imread(projections.loc[i,"file_path"])
        new_projection = oi.imcrop2(new_projection,[0,0,new_width,new_height],value=(150,150,150))
        cropped_projection_path = os.path.join(cropped_projection_folder,os.path.split(projections.loc[i,"file_path"])[1])
        cv2.imwrite(cropped_projection_path,new_projection)
        projections.loc[i,"file_path"] = cropped_projection_path

plot_vars = None

if os.path.isfile(args.annotation_file):
    print("Adding annotation info from: "+args.annotation_file)
    annotation = pd.read_excel(args.annotation_file,sheet_name = "annotation")
    annotation.drop("file_path",axis="columns",inplace=True)
    projections = projections.merge(annotation,on="file_id",how="left")
    
    plot_vars = pd.read_excel(args.annotation_file,sheet_name = "plot_vars")
    
    projections["pdf_file"] = os.path.splitext(os.path.split(args.annotation_file)[1])[0]

else: 
    projections["pdf_file"] = "defaults"
    print("Did not find annotation file: "+args.annotation_file)

print("Making pdf from this info:")
print(projections)

if plot_vars is not None:
    file_vars = ["pdf_file"] + list(plot_vars.loc[plot_vars["plot_axis"]=="file","variable"]) 
    x_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="x","variable"]) 
    y_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="y","variable"]) 
    image_vars = list(plot_vars.loc[plot_vars["plot_axis"]=="image","variable"])
    
    file_vars_sortdirs = [True]
    x_vars_sortdirs = list(plot_vars.loc[plot_vars["plot_axis"]=="x","sort_ascending"])
    y_vars_sortdirs = list(plot_vars.loc[plot_vars["plot_axis"]=="y","sort_ascending"])
    
    sort_directions = file_vars_sortdirs+y_vars_sortdirs+x_vars_sortdirs
    oi.plot_images_pdf(args.projections_pdf_folder,projections,file_vars,x_vars,y_vars,image_vars,image_dim = image_dim_goal,sort_ascending = sort_directions)

else: 
    print("Custom plot vars not available, so using default plotting without annotations")

    oi.plot_images_pdf(args.projections_pdf_folder,projections,file_vars = ["pdf_file"],x_vars = ["channel"],y_vars = ["time","series"],image_vars=["file_id"],image_dim = image_dim_goal)

