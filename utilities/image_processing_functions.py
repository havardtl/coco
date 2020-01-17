import os
import math
import itertools
import subprocess

import pandas as pd
import cv2

import utilities.classes as classes 

IMG_FILE_ENDING = ".ome.tiff"

def print_percentage(i,tot,mod): 
    '''
    Print progress thorugh a for loop 
    
    i    : int : current i
    tot  : int : Maximal i-value
    mod  : int : How much difference is needed before printing
    '''
    step = (1/tot)*100 
    step = step * 0.99
    percent = (i/tot)*100
    if (percent % mod) < step: 
        print("{:.1f}%".format(percent),end="  ",flush=True)
    if i == (tot-1): 
        print("100%")
    
def excel_to_segment_settings(excel_path): 
    '''
    Convert Excel file to list of Segment_settings objects
    
    Params 
    excel_path : str : Path to excel file to read
    
    Returns
    segment_settings : list of Segment_settings : List of Segment_settings
    '''
    
    df_segment_settings = pd.read_excel(excel_path,sheet_name = classes.SEGMENT_SETTINGS_SHEET)
    
    df_segment_settings = df_segment_settings.where((pd.notnull(df_segment_settings)), None) #Convert invalid cells to None instead of NaN

    segment_settings = []
    for channel in df_segment_settings["channel_index"]: 
        s = df_segment_settings.loc[df_segment_settings["channel_index"]==channel,]
        if len(s["channel_index"])!=1:
            raise ValueError("Need at least one unique channel index in segment settings.")
        s = s.iloc[0,]
        segment_settings_c = classes.Segment_settings(channel,s["global_max"],s["color"],s["contrast"],s["auto_max"],s["thresh_type"],s["thresh_upper"],s["thresh_lower"],s["open_kernel"],s["close_kernel"],s["min_size"],s["combine"])
        segment_settings.append(segment_settings_c)
        
    return segment_settings
        
def img_paths_to_zstack_classes(img_paths,segment_settings): 
    '''
    Convert a list of images extracted with bfconvert into Zstack classes. 
    
    Params
    img_paths        : list of str              : Paths to files extracted with bfconvert. Assumes file names according to following convention: {experiment}_{plate}_{time}_{well}_{img_numb}_{other_info}_INFO_{series_index}_{time_index}_{z_index}_{channel_index}{file_ending}
    segment_settings : list of Segment_settings : List of segment settings classes. One for each channel index
    
    Returns
    z_stacks    : list of Zstack : Files organized into Zstack classes for further use
    
    '''
    global IMG_FILE_ENDING

    df_images = pd.DataFrame(data = {"full_path":img_paths})
    df_images["root"],df_images["fname"] = df_images["full_path"].str.rsplit("/",1).str
    df_images["fid"],df_images["info"] = df_images["fname"].str.split("_INFO_",1).str
    df_images["info"] = df_images["info"].str.replace(IMG_FILE_ENDING,"")
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
                global_max = None
                for j in segment_settings: 
                    if str(j.channel_index) == str(this_image.loc[i,"channel_index"]):
                        color = j.color
                        global_max = j.global_max
                        break 
                channel = classes.Channel(this_image.loc[i,"full_path"],this_image.loc[i,"channel_index"],this_image.loc[i,"z_index"],color,global_max)
                channels.append(channel)
            images.append(classes.Image(channels,z_index,segment_settings))
        
        file_id = this_z["fid"].iloc[0]
        time_index = this_z["time_index"].iloc[0]
        series_index = this_z["series_index"].iloc[0]

        z_stacks.append(classes.Zstack(images,file_id,series_index,time_index))
        
    return z_stacks

def img_paths_to_channel_classes(img_paths,segment_settings): 
    '''
    Convert a list of images extracted with bfconvert into a list of Channel objects. 
    
    Params
    img_paths        : list of str              : Paths to files extracted with bfconvert. Assumes file names according to following convention: {file_id}_INFO_{series_index}_{time_index}_{z_index}_{channel_index}{file_ending}
    segment_settings : list of segment_settings : Settings for processing channel 
    
    Returns
    channels    : list of Channel : Files from one z_stack organized into Channel classes 
    '''
    global IMG_FILE_ENDING 
    
    df_images = pd.DataFrame(data = {"full_path":img_paths})
    df_images["root"],df_images["fname"] = df_images["full_path"].str.rsplit("/",1).str
    df_images["fid"],df_images["info"] = df_images["fname"].str.split("_INFO_",1).str
    df_images["info"] = df_images["info"].str.replace(IMG_FILE_ENDING,"")
    df_images["series_index"],df_images["time_index"],df_images["z_index"],df_images["channel_index"] = df_images["info"].str.split("_",3).str

    df_images["image_id"] = "S"+df_images["series_index"]+"T"+df_images["time_index"]+"Z"+df_images["z_index"]
    df_images["z_stack_id"] = "S"+df_images["series_index"]+"T"+df_images["time_index"]
    
    color = None 

    z_stacks = []
    for z_stack_id in [df_images["z_stack_id"].unique()[0]]: 
        this_z = df_images.loc[df_images["z_stack_id"]==z_stack_id,]
        
        channels = []
        for i in this_z.index:
            color = None
            global_max = None
            for j in segment_settings: 
                if str(j.channel_index) == str(this_z.loc[i,"channel_index"]):
                    color = j.color
                    global_max = j.global_max
                    break 
            channel = classes.Channel(this_z.loc[i,"full_path"],this_z.loc[i,"channel_index"],this_z.loc[i,"z_index"],color,global_max) 
            channels.append(channel)
        
    return channels

def choose_z_slices(all_z_slices,to_choose):
    '''
    Extract only some of the z-slices
    
    Params
    all_z_slices : list of int : All available z-slices. Must be pre-sorted
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
        for i in temp:
            i = int(i)
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
    min_size     : list of int   : 
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
    
    df = pd.read_excel(settings_file_path,sheet_name = classes.TEST_SETTINGS_SHEET)
    
    contrast     = excel_column_to_list(df["contrast"])
    auto_max     = excel_column_to_list(df["auto_max"])
    thresh_type  = excel_column_to_list(df["thresh_type"])
    thresh_upper = excel_column_to_list(df["thresh_upper"])
    thresh_lower = excel_column_to_list(df["thresh_lower"])
    open_kernel  = excel_column_to_list(df["open_kernel"])
    close_kernel = excel_column_to_list(df["close_kernel"])
    min_size     = excel_column_to_list(df["min_size"])
    
    return contrast,auto_max,thresh_type,thresh_upper,thresh_lower,open_kernel,close_kernel, min_size

def make_test_settings(load_settings_path,order = None):
    '''
    Params
    load_settings_path  : str         : Path to file specifying ranges of settings to test. 
    order               : list of str : If you provide this value, the order of output variables is sorted according to these.  
    
    Returns
    test_settings : list of Segment_settings : List of all the possible setting combinations as Segment_settings objects
    '''
    
    channel_index = None
    color = None
    combine = False
    
    contrast,auto_max,thresh_type,thresh_upper,thresh_lower,open_kernel,close_kernel,min_size = load_test_settings(load_settings_path)
    
    settings_names = ["contrast","auto_max","thresh_type","thresh_upper","thresh_lower","open_kernel","close_kernel","min_size"]
    df_s = pd.DataFrame(columns=settings_names)
    for combination in itertools.product(contrast,auto_max,thresh_type,thresh_upper,thresh_lower,open_kernel,close_kernel,min_size):
        combination = list(combination)
        temp = pd.Series(combination,index = settings_names)
        df_s = df_s.append(temp,ignore_index = True)
    
    #If binary threshold, upper threshold must be 255
    df_s.loc[df_s["thresh_type"] == "binary","thresh_upper"] = 255 
    
    df_s.drop_duplicates(inplace = True)

    if order is not None: 
        df_s.sort_values(by=order,inplace=True)

    test_settings = []
    for i in df_s.index: 
        setting = classes.Segment_settings(channel_index,None,color,df_s.loc[i,"contrast"],df_s.loc[i,"auto_max"],df_s.loc[i,"thresh_type"],df_s.loc[i,"thresh_upper"],df_s.loc[i,"thresh_lower"],df_s.loc[i,"open_kernel"],df_s.loc[i,"close_kernel"],df_s.loc[i,"min_size"],combine)
        test_settings.append(setting)

    return test_settings

def delete_folder_with_content(folder):
    '''
    Delete folder and all its content. 
    
    Params
    folder : str : path to folder to delete
    '''
    exit_code = os.system('rm -rf '+folder)
    #if exit_code != 0: 
    #    raise ValueError("Could not delete folder: "+folder)

def plot_images_pdf(save_folder,df,file_vars,x_vars,y_vars,image_vars,image_dim,processed_folder,sort_ascending=None,max_size=5000*5000):
    '''
    Create PDFs with images grouped after variables
    
    Params
    df               : pd.Dataframe : Data frame with file paths to images (df["file_path"]) and their variables to sort them by. rows = images, cols = variables
    file_vars        : list of str  : Column names of varibles used to make different PDF files
    x_vars           : list of str  : Column names of variables to plot on x-axis
    y_vars           : list of str  : Column names of variables to plot on y-axis
    image_vars       : list of str  : Column names of variables to print on top of image
    image_dim        : tuple of int : image dimensions
    processed_folder : str          : Path to folder changed images are stored in
    sort_ascending   : list of bool : Directions to sort each column in, file_vars then y_vars then x_vars. Must be equal in length to 1+len(x_vars)+len(y_vars) (file_vars is compressed to 1 column). True = ascending sort, False = descending sort.  
    max_size         : int          : max size in tot pixels of image
    '''
    
    os.makedirs(save_folder,exist_ok=True)

    df["file_vars"] = ""
    sep=""
    for i in file_vars: 
        temp = df[i].astype(str).str.replace("_","")
        df["file_vars"] = df["file_vars"] + sep + temp
        sep="_"
    
    if len(x_vars)==0: 
        x_vars = ["x"]
        df["x"]="None"
        sort_ascending = sort_ascending.append(True)
    df["x_vars"] = ""
    for i in x_vars: 
        temp = df[i].astype(str).str.replace("_","")
        df["x_vars"] = df["x_vars"] +"_"+ temp
    
    if len(y_vars)==0:
        raise ValueErorr("y_vars must be longer than zero")
    
    df["y_vars"] = ""
    for i in y_vars: 
        temp = df[i].astype(str).str.replace("_","")
        df["y_vars"] = df["y_vars"] + "_" + temp
    
    if sort_ascending is None: 
        sort_ascending = [True]*len(["file_vars"]+y_vars+x_vars)
    df.sort_values(by = ["file_vars"]+y_vars+x_vars,ascending=sort_ascending,inplace=True)

    for f in df["file_vars"].unique():
        df_f = df[df["file_vars"]==f].copy()
        xy_positions = pd.DataFrame(index = df_f["y_vars"].unique(),columns = df_f["x_vars"].unique())
        for x_pos in range(0,len(xy_positions.columns)):
            for y_pos in range(0,len(xy_positions.index)):
                xy_positions.iloc[y_pos,x_pos] = (x_pos,y_pos)
        
        df_f["pdf_x_position"] = None 
        df_f["pdf_y_position"] = None 
        for i in df_f.index:
            xy = xy_positions.loc[df_f.loc[i,"y_vars"],df_f.loc[i,"x_vars"]]
            df_f.loc[i,"pdf_x_position"] = xy[0] 
            df_f.loc[i,"pdf_y_position"] = xy[1]
        
        save_path = os.path.join(save_folder,f+".pdf")
        make_pdf(save_path,df_f,x_vars,y_vars,image_vars,processed_folder,image_dim,max_size)

def make_pdf(save_path,df,x_vars,y_vars,image_vars,processed_folder,image_dim,max_size=5000*5000): 
    '''
    Input df with images and make a pdf out of them
    
    Params
    save_path        : str          : Path to save pdf in
    df               : pd.DataFrame : Data frame with paths to images and columns to sort images by
    x_vars           : list of str  : Column names of variables to plot on x-axis
    y_vars           : list of str  : Column names of variables to plot on y-axis
    image_vars       : list of str  : Column names of variables to print on top of image 
    processed_folder : str          : Path to folder changed images are stored in
    image_dim        : tuple of int : Image format in pixels [width,height] that the images has to be in
    max_size         : int          : max size in tot pixels of image
    '''
    
    pdf_imgs = []
    
    for i in df.index: 
        full_path = df.loc[i,"full_path"]
        x = df.loc[i,"pdf_x_position"]
        y = df.loc[i,"pdf_y_position"]
        data = df.loc[i,image_vars+x_vars+y_vars].to_dict()
        pdf_imgs.append(classes.Image_in_pdf(x,y,full_path,data,x_vars,y_vars,image_vars,processed_folder,image_dim,max_size))

    pdf = classes.Pdf(save_path,pdf_imgs)
    pdf.make_pdf()

def imcrop(img,bbox,value=150):
    """
    Crop image with border equal value if it is outside image
    
    Params
    img     : np.array    : cv2 image
    bbox    : list of int : [x1,y1,x2,y2]
    value   :             : color of added border

    Returns
    img     : np.array    : cv2 image with border 
    """
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2,value=value)
    return img[y1:y2, x1:x2]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2,value=255):
    """
    Add border to image and return modified coordinates after border addition
    
    Params
    img             : numpy array   : image
    x1,x2,y1,y2:    : int           : coordinates
    Returns
    img             : numpy array   : image
    x1,x2,y1,y2:    : int           : coordinates after padding
    """
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0), -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_CONSTANT,value=value)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def img_dim_str_to_tuple(img_dim_str): 
        #image dimensions stored as string version of tupple in dataframe, convert it to tupple int here
        img_dim_str = img_dim_str.replace("(","").replace(")","")
        img_dim = img_dim_str.split(", ",1)
        img_dim = (int(img_dim[0]),int(img_dim[1]))
        return img_dim


def all_images_in_same_ratio(df,cropped_images_folder):
    '''
    Crop all images so that they have the same ratio. Uses the most common ratio as base. 
    
    Params
    df                    : pd.DataFrame : Data frame with images to plot. Needs to contain "full_path" column with paths to images and "img_dim" with image dimensions as string representations of int-tupples: str((int,int)). 
    cropped_images_folder : str : Path to folder with cropped images
    
    Returns
    df                    : pd.DataFrame : Data frame where "full_path" column has gotten updated paths to cropped images
    '''
    
    most_common_img_dim = str(df["img_dim"].mode()[0])
    image_dim_goal = img_dim_str_to_tuple(most_common_img_dim)
    image_dim_goal = (int(image_dim_goal[0]),int(image_dim_goal[1]))
    goal_ratio = float(image_dim_goal[1])/float(image_dim_goal[0])

    for i in df.index:
        img_dim = df.loc[i,"img_dim"]
        if img_dim != most_common_img_dim:
            img_dim = img_dim_str_to_tuple(img_dim)
            img_dim_ratio = float(img_dim[1])/float(img_dim[0])
            
            if img_dim_ratio < goal_ratio:
                new_width = int((img_dim_ratio/goal_ratio)*float(img_dim[0]))
                new_height = int(img_dim[0])
            else:
                new_height  = int(img_dim[0])
                new_width = int(new_height * 1/goal_ratio)
                new_ratio = new_height/new_width

            new_projection = cv2.imread(df.loc[i,"full_path"])
            new_projection = imcrop(new_projection,[0,0,new_height,new_width],value=(150,150,150))
            cropped_projection_path = os.path.join(cropped_images_folder,os.path.split(df.loc[i,"full_path"])[1])
            cv2.imwrite(cropped_projection_path,new_projection)
            df.loc[i,"full_path"] = cropped_projection_path
    
    return df

