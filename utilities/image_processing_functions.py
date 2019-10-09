import os
import math
import itertools

import pandas as pd
import cv2

import utilities.classes as classes

def get_images(file_path,extract_to_folder,info_file="files_info.txt",bfconvert_info_str = "_INFO_%s_%t_%z_%c",file_ending = ".ome.tiff"):
    '''
    Get individual stack images from microscopy format file

    Params:
    file_path           : str   : path to the file 
    extract_to_folder   : str   : folder to store the extracted files. Log file stored one folder above.
    info_file           : str   : path to csv file with data about files  
    bfconvert_info_str  : str   : string added to end of filen name of resulting files. See bfconvert for what special symbols are converted to. 
    file_ending         : str   : file format to save output file in
     
    Returns
    img_paths         : list of str : paths to extracted files. 
    '''
    
    os.makedirs(extract_to_folder,exist_ok=True)
    
    fname = os.path.splitext(os.path.split(file_path)[1])[0]
    out_name = fname+bfconvert_info_str+file_ending
    out_name = os.path.join(extract_to_folder,out_name)
    log_file = os.path.join(os.path.split(extract_to_folder)[0],"log_"+fname+".txt")

    cmd = "bfconvert -overwrite \"{file_path}\" \"{out_name}\" > {log_file}".format(file_path = file_path,out_name = out_name,log_file=log_file)

    #print(cmd)
    exit_val = os.system(cmd)
    if exit_val !=0:
        raise RuntimeError("This command did not exit properly: "+cmd)
    
    img_paths = [] 
    
    for path in os.listdir(extract_to_folder):
        if path.endswith(file_ending):
            img_paths.append(os.path.join(extract_to_folder,path))
    
    with open(os.path.join(extract_to_folder,info_file),'w') as f: 
        for path in img_paths: 
            f.write(path+"\n")
    
    return img_paths

def img_paths_to_zstack_classes(img_paths,file_ending,segment_settings): 
    '''
    Convert a list of images extracted with bfconvert into Zstack classes. 
    
    Params
    img_paths        : list of str              : Paths to files extracted with bfconvert. Assumes file names according to following convention: {experiment}_{plate}_{time}_{well}_{img_numb}_{other_info}_INFO_{series_index}_{time_index}_{z_index}_{channel_index}{file_ending}
    file_ending      : str                      : File ending of file 
    segment_settings : list of Segment_settings : List of segment settings classes. One for each channel index
    
    Returns
    z_stacks    : list of Zstack : Files organized into Zstack classes for further use
    
    '''
    df_images = pd.DataFrame(data = {"full_path":img_paths})
    df_images["root"],df_images["fname"] = df_images["full_path"].str.rsplit("/",1).str
    df_images["fid"],df_images["info"] = df_images["fname"].str.split("_INFO_",1).str
    df_images["experiment"],df_images["plate"],df_images["time"],df_images["well"],df_images["img_numb"],df_images["other_info"] = df_images["fid"].str.split("_",5).str
    df_images["info"] = df_images["info"].str.replace(file_ending,"")
    df_images["series_index"],df_images["time_index"],df_images["z_index"],df_images["channel_index"] = df_images["info"].str.split("_",3).str

    df_images["image_id"] = "S"+df_images["series_index"]+"T"+df_images["time_index"]+"Z"+df_images["z_index"]
    df_images["z_stack_id"] = "S"+df_images["series_index"]+"T"+df_images["time_index"]

    z_stacks = []
    for z_stack_id in df_images["z_stack_id"].unique(): 
        this_z = df_images.loc[df_images["z_stack_id"]==z_stack_id,]
        
        images = []
        for z_index in this_z["z_index"].unique():
            this_image = this_z.loc[this_z["z_index"]==z_index,]
            channels = []
            for i in this_image.index: 
                color = None
                for j in segment_settings: 
                    if str(j.channel_index) == str(this_image.loc[i,"channel_index"]):
                        color = j.color
                        break 
                channel = classes.Channel(this_image.loc[i,"full_path"],this_image.loc[i,"channel_index"],this_image.loc[i,"z_index"],color)
                channels.append(channel)
            images.append(classes.Image(channels,z_index,segment_settings))
        
        experiment = this_z["experiment"].iloc[0]
        plate = this_z["plate"].iloc[0]
        time = this_z["time"].iloc[0]
        well = this_z["well"].iloc[0]
        img_numb = this_z["img_numb"].iloc[0]
        other_info = this_z["other_info"].iloc[0]
        time_index = this_z["time_index"].iloc[0]
        series_index = this_z["series_index"].iloc[0]

        z_stacks.append(classes.Zstack(images,experiment,plate,time,well,img_numb,other_info,series_index,time_index))
        
    return z_stacks


def img_paths_to_channel_classes(img_paths,file_ending): 
    '''
    Convert a list of images extracted with bfconvert into a list of Channel objects. 
    
    Params
    img_paths        : list of str              : Paths to files extracted with bfconvert. Assumes file names according to following convention: {experiment}_{plate}_{time}_{well}_{img_numb}_{other_info}_INFO_{series_index}_{time_index}_{z_index}_{channel_index}{file_ending}
    file_ending      : str                      : File ending of file 
    
    Returns
    channels    : list of Channel : Files from one z_stack organized into Channel classes 
    '''
    df_images = pd.DataFrame(data = {"full_path":img_paths})
    df_images["root"],df_images["fname"] = df_images["full_path"].str.rsplit("/",1).str
    df_images["fid"],df_images["info"] = df_images["fname"].str.split("_INFO_",1).str
    df_images["experiment"],df_images["plate"],df_images["time"],df_images["well"],df_images["img_numb"],df_images["other_info"] = df_images["fid"].str.split("_",5).str
    df_images["info"] = df_images["info"].str.replace(file_ending,"")
    df_images["series_index"],df_images["time_index"],df_images["z_index"],df_images["channel_index"] = df_images["info"].str.split("_",3).str

    df_images["image_id"] = "S"+df_images["series_index"]+"T"+df_images["time_index"]+"Z"+df_images["z_index"]
    df_images["z_stack_id"] = "S"+df_images["series_index"]+"T"+df_images["time_index"]
    
    color = None 

    z_stacks = []
    for z_stack_id in [df_images["z_stack_id"].unique()[0]]: 
        this_z = df_images.loc[df_images["z_stack_id"]==z_stack_id,]
        
        channels = []
        for i in this_z.index: 
            channel = classes.Channel(this_z.loc[i,"full_path"],this_z.loc[i,"channel_index"],this_z.loc[i,"z_index"],color)
            channels.append(channel)
        
    return channels


def choose_z_slices(all_z_slices,to_choose):
    '''
    Extract only some of the z-slices
    
    Params
    all_z_slices : list of int : All available z-slices
    to_choose    : str         : which z_slice to choose. "1" = 1, "1:3" = [1,2,3], "1,5,8" = [1,5,8] and "i3" = 3 evenly choosen from range.

    Return
    choosen_z_slices : list of str : z_slices to process
    '''
    choosen_z_slices = []
    if "i" in to_choose:
        i_to_keep = int(to_choose.replace("i",""))
        step_size = math.ceil(len(all_z_slices)/i_to_keep)

        for i in range(0,len(all_z_slices),step_size):
            choosen_z_slices.append(all_z_slices[i])
    elif ":" in to_choose: 
        first,last = to_choose.split(":",1)
        to_choose = list(range(int(first),int(last)))
        
    elif "," in to_choose:
        temp = to_choose.split(",")
        for i in range(len(temp)):
            choosen_z_slices.append(all_z_slices[i])
    else:
        i = int(to_choose)
        choosen_z_slices.append(all_z_slices[i])

    return choosen_z_slices
    
def load_test_settings(settings_file_path):
    '''
    Read a file specifying the test settings to test
    
    Params
    settings_file_path : str : Path to file to read
    
    Returns
    contrast     : list of float : 
    auto_max     : list of bool  : 
    thresh_type  : list of str   : 
    thresh_upper : list of int   :
    thresh_lower : list of int   : 
    open_kernel  : list of int   : 
    close_kernel : list of int   : 
    '''
    
    def excel_column_to_list(column):
        column = column.drop_duplicates()
        out = []
        for c in column: 
            if not pd.isna(c):
                if c == "None": 
                    c = None
                out.append(c)
        return out 
    
    df = pd.read_excel(settings_file_path)
    
    contrast     = excel_column_to_list(df["contrast"])
    auto_max     = excel_column_to_list(df["auto_max"])
    thresh_type  = excel_column_to_list(df["thresh_type"])
    thresh_upper = excel_column_to_list(df["thresh_upper"])
    thresh_lower = excel_column_to_list(df["thresh_lower"])
    open_kernel  = excel_column_to_list(df["open_kernel"])
    close_kernel = excel_column_to_list(df["close_kernel"])
    
    return contrast,auto_max,thresh_type,thresh_upper,thresh_lower,open_kernel,close_kernel  

def make_test_settings(load_settings_path):
    '''
    Params
    load_settings_path  : str : Path to file specifying ranges of settings to test. 
    
    Returns
    test_settings : list of Segment_settings : List of all the possible setting combinations as Segment_settings objects
    '''
    
    channel_index = None
    color = None
    combine = False
    
    contrast,auto_max,thresh_type,thresh_upper,thresh_lower,open_kernel,close_kernel = load_test_settings(load_settings_path)
    
    settings_names = ["contrast","auto_max","thresh_type","thresh_upper","thresh_lower","open_kernel","close_kernel"]
    df_s = pd.DataFrame(columns=settings_names)
    for combination in itertools.product(contrast,auto_max,thresh_type,thresh_upper,thresh_lower,open_kernel,close_kernel):
        combination = list(combination)
        temp = pd.Series(combination,index = settings_names)
        df_s = df_s.append(temp,ignore_index = True)
    
    #If binary threshold, upper threshold must be 255
    df_s.loc[df_s["thresh_type"] == "binary","thresh_upper"] = 255 
    
    df_s.drop_duplicates(inplace = True)

    test_settings = []
    for i in df_s.index: 
        setting = classes.Segment_settings(channel_index,color,df_s.loc[i,"contrast"],df_s.loc[i,"auto_max"],df_s.loc[i,"thresh_type"],df_s.loc[i,"thresh_upper"],df_s.loc[i,"thresh_lower"],df_s.loc[i,"open_kernel"],df_s.loc[i,"close_kernel"],combine)
        test_settings.append(setting)

    return test_settings
    
def make_channel_masks(channels,setting,mask_save_folder,setting_id): 
    '''
    Make masks for all channels with the specified mask creation settings
    
    Params
    channels   : list of Channel  : Channels that needs mask to be created
    mask_save_folder : str              : Folder to save masks in. Same length as channels. 
    setting    : Segment_settings : Segmentation settings for making mask
    setting_id : str              : setting_id to print when running function
    '''
    print(setting_id,end="\t",flush=True)

    save_paths = []
    for c in channels: 
        save_paths.append(os.path.join(mask_save_folder,c.file_id+"_setting-"+str(setting_id)+".png"))
    
    for i in range(len(channels)): 
        channels[i].make_mask(setting)
        mask = channels[i].get_mask()
        cv2.imwrite(save_paths[i],mask)
    
    return save_paths












