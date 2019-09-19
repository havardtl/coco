#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Make annotation file for entering data about images.')
parser.add_argument('--raw_data',default="rawdata",type=str,help='Path to raw image. File names must follow this convention: "experiment_plate_day_well_object_otherinfo.*" Can use all formats accepted by your installed version of bftools. ')
parser.add_argument('--annotation_input',default="exp_setup.txt",type=str,help='Looks for annotation file that contains this string in file name in all above folders. Excludes all info above "--Sample ID--" if that string is present. Adds annotation from it automatically by looking for "well_id" from this table in raw file name.')
parser.add_argument('--well_id_remove',default = r'_d[0-9][0-9]',type=str,help='Remove this regular expression from file name before trying to match "well_id"')
parser.add_argument('--annotation_file',type=str,default = 'annotation1.xlsx', help='Output file containing annotation data')
args = parser.parse_args()

########################
# Imports 
########################
import os
from io import StringIO 
import re
import pandas as pd

########################
# Script 
########################
raw_files = []
file_id = []
for root, dirs, files in os.walk(args.raw_data):
    for f in files: 
        raw_files.append(os.path.join(root,f))
        file_id.append(os.path.splitext(f)[0])
        
if len(raw_files)<1:
    raise ValueError("Did not find any raw files in "+args.raw_data)

df = pd.DataFrame(data = {"file_path":raw_files,"file_id":file_id,"treatment":None,"is_control":False,"stain_group":None})

# Find exp setup file
path_to_here = os.path.abspath(".")

folder = path_to_here
exp_setup_file = None
while folder.count("/")>1 and not exp_setup_file is not None: 
    for f in os.listdir(folder):
        if args.annotation_input in f:
            exp_setup_file = os.path.join(folder,f)
    folder = os.path.split(folder)[0]

if exp_setup_file is not None:
    print("Adding info from: "+exp_setup_file)
    #Process exp_setup file and read to data frame
    with open(exp_setup_file) as f: 
        exp_setup_raw = f.read()
    
    #Exlude everything above "--Sample ID--" if string is present
    sample_info_loc = exp_setup_raw.find("--Sample ID--")
    if sample_info_loc >= 0: 
        sample_info = exp_setup_raw[sample_info_loc:]
        first_line = sample_info.find("\n")
        sample_info = sample_info[(first_line+1):]
    
    sample_info = sample_info.strip()
    sample_info = sample_info.replace(";",",")
    if sample_info.count(",")<=(sample_info.count("\n")-1):
        while "  " in sample_info: 
            sample_info = sample_info.replace("  "," ")
        while "\t\t" in sample_info: 
            sample_info = sample_info.replace("\t\t","\t")
        sample_info = sample_info.replace(" ",",")
        sample_info = sample_info.replace("\t",",")

    exp_stringio = StringIO(sample_info)
    
    exp_df = pd.read_csv(exp_stringio,sep=",")
    
    df["exp"],df["plate"],df["day"],df["well"],df["object"],df["other_info"] = df["file_id"].str.split("_",5).str
    
    df["well_id"] = df["exp"]+"_"+df["plate"]+"_"+df["well"]
    
    if "treatment" in exp_df.columns:
        df.drop("treatment",axis="columns",inplace=True)
    if "stain_group" in exp_df.columns:
        df.drop("stain_group",axis="columns",inplace=True)
    if "is_control" in exp_df.columns:
        df.drop("is_control",axis="columns",inplace=True)
    
    df = df.merge(exp_df,on="well_id",how="left")
print(df)
#Make sheet with plotting information for all variables
always_present = ["channel","time" ,"series","img_dim"]
plot_axis      = ["x"      ,"y"    ,"y"     ,None     ]
importance     = [1        ,1      ,2       ,0        ]
in_annotation  = list(df.columns)

plot_vars = pd.DataFrame(data = {"variable":always_present,"plot_axis":plot_axis,"importance":importance,"sort_ascending":True})
in_annotation = pd.DataFrame(data = {"variable":list(df.columns),"plot_axis":None,"importance":0,"sort_ascending":True})
plot_vars = pd.concat([plot_vars,in_annotation])

plot_vars.loc[plot_vars["variable"]=="file_id","plot_axis"] = "image" 
plot_vars.loc[plot_vars["variable"]=="file_id","importance"] = 0 
plot_vars.loc[plot_vars["variable"]=="treatment","plot_axis"] = "y" 
plot_vars.loc[plot_vars["variable"]=="treatment","importance"] = 3 
plot_vars.loc[plot_vars["variable"]=="is_control","plot_axis"] = "y" 
plot_vars.loc[plot_vars["variable"]=="is_control","importance"] = 4 
plot_vars.loc[plot_vars["variable"]=="is_control","sort_ascending"] = False 

plot_vars.sort_values(by = ["plot_axis","importance","variable"],ascending=False,axis="index",inplace=True)

with pd.ExcelWriter(args.annotation_file) as writer: 
    df.to_excel(writer,sheet_name="annotation")
    plot_vars.to_excel(writer,sheet_name="plot_vars")

print("Wrote annotation info to: "+args.annotation_file)

