import os

import pandas as pd

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
    
