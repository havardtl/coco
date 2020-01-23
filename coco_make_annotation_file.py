#!/usr/bin/env python3

#######################
# Argument parsing
#######################
import argparse
parser = argparse.ArgumentParser(description = 'Make annotation file for entering data about images.')
parser.add_argument('--raw_data',default="raw/rawdata",type=str,help='Path to raw image. File names must follow this convention: "experiment_plate_day_well_object_otherinfo.*" Can use all formats accepted by your installed version of bftools. ')
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

import utilities.classes as classes

##########################
# Functions
##########################
def strip_lines(string): 
    #convert string into lines and then strip each line and convert string back to multi-line string again
    out = ""
    for l in string.splitlines(): 
        out = out + l.strip() + "\n"
    return out

########################
# Script 
########################
print("Making annotation file: "+args.annotation_file)

file_id = os.listdir(args.raw_data)
raw_files = []
for i in range(len(file_id)): 
    raw_files.append(os.path.join(args.raw_data,file_id[i]))
    file_id[i] = os.path.splitext(file_id[i])[0]
        
if len(file_id)<1:
    raise ValueError("Did not find any raw files in "+args.raw_data)

df = pd.DataFrame(data = {"full_path":raw_files,"file_id":file_id,"treatment":None,"replicate":None,"is_control":False,"stain_group":None})

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
    
    #Convert sample_info from tabular or white space format to csv format 
    sample_info = strip_lines(sample_info)
    sample_info = sample_info.replace(";",",")
    if sample_info.count(",")<=(sample_info.count("\n")-1):
        sample_info = sample_info.replace(" ",",")
        sample_info = sample_info.replace("\t",",")
        while ",," in sample_info: 
            sample_info = sample_info.replace(",,",",")
    
    exp_stringio = StringIO(sample_info)
    
    exp_df = pd.read_csv(exp_stringio,sep=",")
    
    df["exp"],df["plate"],df["day"],df["well"],df["img_numb"],df["other_info"] = df["file_id"].str.split("_",5).str
    
    df["well_id"] = df["exp"]+"_"+df["plate"]+"_"+df["well"]
    
    #Replacing default columns if they are present in experiment info
    if "treatment" in exp_df.columns:
        df.drop("treatment",axis="columns",inplace=True)
    if "replicate" in exp_df.columns:
        df.drop("replicate",axis="columns",inplace=True)
    if "stain_group" in exp_df.columns:
        df.drop("stain_group",axis="columns",inplace=True)
    if "is_control" in exp_df.columns:
        df.drop("is_control",axis="columns",inplace=True)
    print(exp_df)
    df = df.merge(exp_df,on="well_id",how="left")
    
print(df)

#Make sheet with plotting information for all variables
always_present = ["id_channel","channel_index","time_index" ,"series_index","img_dim"]
plot_axis      = ["x"         , None      ,"y"    ,"y"     ,None     ]
importance     = [1           ,0          ,0      ,2       ,0        ]
in_annotation  = list(df.columns)

plot_vars = pd.DataFrame(data = {"variable":always_present,"plot_axis":plot_axis,"importance":importance,"sort_ascending":True})
in_annotation = pd.DataFrame(data = {"variable":list(df.columns),"plot_axis":None,"importance":0,"sort_ascending":True})
plot_vars = pd.concat([plot_vars,in_annotation])

plot_vars.loc[plot_vars["variable"]=="full_path","plot_axis"] = "image" 
plot_vars.loc[plot_vars["variable"]=="full_path","importance"] = 1
plot_vars.loc[plot_vars["variable"]=="file_id","plot_axis"] = "y" 
plot_vars.loc[plot_vars["variable"]=="file_id","importance"] = 3 
plot_vars.loc[plot_vars["variable"]=="treatment","plot_axis"] = "y" 
plot_vars.loc[plot_vars["variable"]=="treatment","importance"] = 3 
plot_vars.loc[plot_vars["variable"]=="is_control","plot_axis"] = "y" 
plot_vars.loc[plot_vars["variable"]=="is_control","importance"] = 4 
plot_vars.loc[plot_vars["variable"]=="is_control","sort_ascending"] = False 

plot_vars.sort_values(by = ["plot_axis","importance","variable"],ascending=False,axis="index",inplace=True)

#Import defaults from DEFAULT_SETTINGS_XLSX
test_settings    = pd.read_excel(classes.DEFAULT_SETTINGS_XLSX,sheet_name = classes.TEST_SETTINGS_SHEET)
segment_settings = pd.read_excel(classes.DEFAULT_SETTINGS_XLSX,sheet_name = classes.SEGMENT_SETTINGS_SHEET)

with pd.ExcelWriter(args.annotation_file) as writer: 
    df.to_excel(writer,sheet_name=classes.ANNOTATION_SHEET,index=False)
    plot_vars.to_excel(writer,sheet_name=classes.PLOT_VARS_SHEET,index=False)
    test_settings.to_excel(writer,sheet_name=classes.TEST_SETTINGS_SHEET,index=False)
    segment_settings.to_excel(writer,sheet_name=classes.SEGMENT_SETTINGS_SHEET,index=False)

print("Wrote annotation info to: "+args.annotation_file)

