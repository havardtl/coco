import os
import math
import itertools
import multiprocessing as mp 
import numpy as np
import pandas as pd
import cv2
import warnings

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader

import utilities.general_functions as oc

def get_images(file_path,extract_to_folder,info_file="files_info.csv"):
    '''
    Get individual stack images from microscopy format file

    Params:
    file_path         : str   : path to the file 
    extract_to_folder : str   : folder to store the extracted files. Log file stored one folder above.
    info_file         : str   : path to csv file with data about files  
    '''
    
    os.makedirs(extract_to_folder,exist_ok=True)
    
    bfconvert_info_string = "_INFO_S%s_C%c_W%w_Z%z_T%t"
    file_ending = ".ome.tiff"
    
    fname = os.path.splitext(os.path.split(file_path)[1])[0]
    out_name = fname+bfconvert_info_string+file_ending
    out_name = os.path.join(extract_to_folder,out_name)
    log_file = os.path.join(os.path.split(extract_to_folder)[0],"log_"+fname+".txt")

    cmd = "bfconvert -overwrite \"{file_path}\" \"{out_name}\" > {log_file}".format(file_path = file_path,out_name = out_name,log_file=log_file)

    #print(cmd)
    exit_val = os.system(cmd)
    if exit_val !=0:
        raise ValueError("This command did not exit properly: "+cmd)
    
    df = oc.walk_to_df(extract_to_folder)
    df["info"]= df["file"].str.split("_INFO_", n = 1, expand = True).iloc[:,1]
    df["info"]= df["info"].str.replace(file_ending,"")
    temp = df["info"].str.split('_',n=4,expand = True)
    temp.columns = ['series_index','channel_index','channel_name','Z_index','T_index']
    
    df = df.join(temp)
    
    df = df[df['file'].str.endswith(file_ending)]
    
    df.to_csv(os.path.join(extract_to_folder,"files_info.csv"))

    return df 

def make_test_settings_df(images_paths,temp_mask_folder):
    '''
    #TODO: make it possible to enter file with settings ranges so that other settings can be used
    
    Params
    images_paths     : list of strings  : Paths to images that settings shall be applied to
    temp_mask_folder : string           : Path to temporary folder for saving images in 
    Returns
    test_settings : pd.DataFrame : Data frame with all of the possible setting combinations. Each column is one setting, each row is one combination 

    '''

    use_defaults = True 
    if use_defaults: 
        shrink = [0.5]
        contrast = [1,1.5]
        auto_max = [None]
        thresh_type = ["canny_edge","binary"]
        thresh_upper = [100,150,200]
        thresh_lower = [50,100]
        open_kernel = [None,np.ones((5)).astype('uint8')]
        close_kernel = [None,np.ones((5)).astype('uint8')]

    settings_names = ["image_path","shrink","contrast","auto_max","thresh_type","thresh_upper","thresh_lower","open_kernel","close_kernel"]
    test_settings = pd.DataFrame(columns=settings_names)
    for combination in itertools.product(images_paths,shrink,contrast,auto_max,thresh_type,thresh_upper,thresh_lower,open_kernel,close_kernel):
        combination = list(combination)
        temp = pd.Series(combination,index = settings_names)
        test_settings = test_settings.append(temp,ignore_index = True)
    
    test_settings["processed"] = True

    unprocessed = pd.DataFrame(columns = settings_names)
    unprocessed["image_path"] = images_paths
    unprocessed.loc[unprocessed.index,"auto_max"] = True
    unprocessed["processed"] = False
    
    test_settings = pd.concat([test_settings,unprocessed],ignore_index=True)
    unprocessed.loc[unprocessed.index,"auto_max"] = False
    test_settings = pd.concat([test_settings,unprocessed],ignore_index=True)

    save_names = []
    for i in test_settings.index:
        save_names.append(os.path.join(temp_mask_folder,str(i)+".png"))

    test_settings["mask_path"] = save_names 

    return test_settings

def get_processed_mask(image_path,shrink,contrast,auto_max,thresh_type, thresh_upper, thresh_lower,open_kernel,close_kernel,print_info=None):
    '''
    Generate one of the processed images with the specified settings

    Params
    image_path  :   str       : Path to image to microscopy image file to process
    print_info  :   str       : Info to print to screen while processing image
    shrink      :   float     : Shrink image to save processing time. e.g. shrink = 0.5 reduces size by half
    contrast    :   float     : Contrast to apply to image before making 8 bit 
    thresh_type :   str       : Threshold type. Possible types = ["canny_edge","binary"] 
    thresh_upper:   int       : Max value in threshold
    thresh_lower :   int       : Lower threshold value. Only relevant for thresh_type = "canny_edge", otherwise set to 0. 
    open_kernel :   np.array  : Image kernel used for removing noise. e.g. np.ones((3,3).np.uint8)
    close_kernel:   np.array  : Image kernel used for closing holes in contours. e.g. np.ones((3,3),np.uint8)
    save_path   :   str       : Path to save mask to. If None it does not save path
    
    Returns 
    mask        :   np.array  : Mask of image after processing  
    
    '''
    if not print_info is None: 
        print(print_info+"\t getting mask from: "+image_path)
    
    img = cv2.imread(image_path,cv2.IMREAD_ANYDEPTH)
    if img is None: 
        raise ValueError("Could not read image: "+image_path)
    
    if (shrink !=1) and (shrink is not None) and (not np.isnan(shrink)):
        new_dim = (int(img.shape[0]*shrink),int(img.shape[1]*shrink))
        img = cv2.resize(img,new_dim,interpolation = cv2.INTER_AREA) 
    
    if (contrast !=1) and contrast is not None and not np.isnan(contrast):
        img = (img*contrast).astype('uint8')
    
    if auto_max is not None and not np.isnan(auto_max): 
        if auto_max: 
            max_pixel = np.max(np.amax(img,axis=0))
            img = (img/(max_pixel/255)).astype('uint8')
    
    thresh_type = str(thresh_type)
    if thresh_type is not None and not (str(thresh_type)+" " == "nan "): 
        if thresh_type == "canny_edge": 
            img = cv2.Canny(img,thresh_lower,thresh_upper)
        elif thresh_type == "binary": 
            ret,img = cv2.threshold(img,thresh_lower,255,cv2.THRESH_BINARY)
        else:
            raise ValueError(str(thresh_type)+" is not an available threshold method")

        if type(open_kernel) is np.ndarray:  
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, open_kernel)
        
        if type(close_kernel) is np.ndarray:  
            img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,close_kernel)
    
    return img

def add_all_channels(df):
    '''
    For every image add the image path to all the corresponding channels. These channels are used for quantitation later. Assumes max 5 channels.

    Params
    df  : pd.DataFrame : Data frame with file information about each file for one aquisition. Need columns: ["file","series_index","Z_index","T_index","channel_index"]

    Returns
    df  : pd.DataFrame : The same data frame as input just with all correlating channels added as columns from channel_0 --> channel_n
    '''
    channel_column_string = "channel_path_"


    df["info_no_channel"] = df["series_index"]+"_"+df["Z_index"]+"_"+df["T_index"]
    
    all_channels = df["channel_index"].unique()
    if len(all_channels)>6: 
        raise ValueError("Number of channels is larger than six. Aborting as this seems unlikely.")
    
    for i in all_channels:
        channel_column = channel_column_string+str(i)
        df[channel_column] = None
    
    for i in df.index:
        this_time_point = df.loc[df.loc[i,"info_no_channel"]==df["info_no_channel"],]
        for j in this_time_point.index:
            channel_column = channel_column_string + this_time_point.loc[j,"channel_index"]
            df.loc[i,channel_column] = this_time_point.loc[j,"file"]
    
    return df 

def make_mask(image_series,segment_settings,mask_save_folder):
    '''
    Make a mask of an image that is saved for later use

    Params
    image_series     : pd.Series    : pandas series with path and other information about image to be processed. Need ["root","file"] for image path and channel paths in column like "channel_path_*" where * is wildcard
    segment_settings : pd.DataFrame : Data frame with segmentation settings. rows = channels, columns = setting. See oi.get_processed_mask for settings needed.
    mask_save_folder : str          : Path to file path for saving path. If set to None no masks are saved

    Returns
    mask_save_path   : str          : Path where mask is saved  
    '''
    image_id = os.path.splitext(os.path.split(image_series["file"])[1])[0]
    mask_save_path = os.path.join(mask_save_folder,image_id+".png")
    if os.path.exists(mask_save_path):
        return mask_save_path
    
    image_path = os.path.join(image_series["root"],image_series["file"])
    
    segment_settings = segment_settings.loc[segment_settings["channel_index"]==image_series["channel_index"],].iloc[0,]
    shrink = segment_settings["shrink"]
    contrast = segment_settings["contrast"]
    auto_max = segment_settings["auto_max"]
    thresh_type = segment_settings["thresh_type"]
    thresh_upper = segment_settings["thresh_upper"]
    thresh_lower = segment_settings["thresh_lower"]
    open_kernel = segment_settings["open_kernel"]
    close_kernel = segment_settings["close_kernel"]

    mask = get_processed_mask(image_path,shrink,contrast,auto_max,thresh_type,thresh_upper,thresh_lower,open_kernel,close_kernel)
    cv2.imwrite(mask_save_path,mask)

    return mask_save_path
 
def get_rois(image_series):
    '''
    Get all the rois in the image
    
    Params
    image_series     : pd.Series  : pandas series with path to image and other information about image to be processed. Need ["root","file"] for image path and channel paths in column like "channel_path_*" where * is wildcard
    previous_z_path  : str        : Path to mask of previous z_plane
    next_z_path      : str        : Path to maks of next z_plane
    '''
    mask = cv2.imread(image_series["mask_path"],cv2.IMREAD_GRAYSCALE)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    channel_dict = {}
    for i in image_series.index:
        if "channel_path_" in i:
            channel_index = i.replace("channel_path_","")
            if image_series[i] is not None: 
                channel_dict[channel_index] = os.path.join(image_series["root"],image_series[i])
            else: 
                channel_dict[channel_index] = None
    
    df = get_shape_info(contours,mask,channel_dict,image_series["shrink"])
    for i in image_series.index: 
        df[i] = image_series[i]
    
    return df

def get_xyz_res(ome_tiff_path,xml_folder):
    '''
    Find physical z_resolution from ome.tiff file

    Params
    ome_tiff_path : str : Path to ome.tiff file to extract information from

    Returns
    physical_size_x : float : x resolution of each pixel in um
    physical_size_y : float : y resolution of each pixel in um
    physical_size_z : float : z resolution of each pixel in um
    '''
    file_id = os.path.split(ome_tiff_path)[1]
    if "ome.tiff" not in ome_tiff_path: 
        raise ValueError("Can only find xyz resolution from ome.tiff files, not from file: "+ome_tiff_path)
    
    xml_path = os.path.join(xml_folder,file_id+".xml")

    if not os.path.exists(xml_path):
        cmd = "tiffcomment {image} > {xml}".format(image = ome_tiff_path,xml = xml_path)
        print(cmd)
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise ValueError("Command did not exit properly"+cmd)


    physical_size_x = None 
    unit_x = None 
    physical_size_y = None  
    unit_y = None 
    physical_size_z = None  
    unit_z = None

    with open(xml_path) as f: 
        xml = f.read()
    xml = xml.split(">")
    for field in xml:
        if "<Pixels " in field:
            field = field.replace("<Pixels ","")
            values = field.split(" ")
            for v in values: 
                name,value = v.split("=")
                value = value.replace("\"","")
                if name == "PhysicalSizeX":
                    physical_size_x = float(value)

                if name == "PhysicalSizeY":
                    physical_size_y = float(value)

                if name == "PhysicalSizeZ":
                    physical_size_z = float(value)
                
                if name == "PhysicalSizeXUnit":
                    unit_x = value
                
                if name == "PhysicalSizeYUnit":
                    unit_y = value
                
                if name == "PhysicalSizeZUnit":
                    unit_z = value

    #print("x: {x} {x_u}\ty: {y} {y_u} \t z: {z} {z_u}".format(x=physical_size_x,x_u=unit_x,y=physical_size_y,y_u=unit_y,z=physical_size_z,z_u=unit_z))
    
    supported_units = ["µm"]
    
    if unit_x not in supported_units:
        raise ValueError("unit_x = "+unit_x+" is not a supported unit. Supported units: "+str(supported_units))
    if unit_y not in supported_units:
        raise ValueError("unit_y = "+unit_y+" is not a supported unit. Supported units: "+str(supported_units))
    if unit_z not in supported_units:
        raise ValueError("unit_x = "+unit_z+" is not a supported unit. Supported units: "+str(supported_units))

    return physical_size_x,physical_size_y,physical_size_z

def build_objects(df_rois):
    '''
    Connect all objects from the same channel and make a df that describes these
    
    Params
    df_rois    : pd.DataFrame   : data frame with all roi information
    '''

    col_names = ["object_id","z_stack_id","root","file","contour_ids","contours_centers_xyz","at_edge","volume"]
    channels = df_rois["channel_index"].unique()
    
    channel_sums_cols = []
    is_inside_cols = []
    for c in channels: 
        channel_sums_cols.append("sum_grey_"+c)
        is_inside_cols.append("is_inside_"+c)

    col_names = col_names + channel_sums_cols + is_inside_cols 

    objects = pd.DataFrame(columns = col_names)

    df_rois["object_id"] = None

    def add_all_contours(all_overlapping,contour_i):
        overlapping = df_rois.loc[contour_i,"overlapping_z"]
        
        #pandas don't support lists in cells very well, so they had to be converted to string. Her I convert back
        if overlapping is not None: 
            if "," in overlapping: 
                overlapping = overlapping.split(",")
                for i in range(len(overlapping)):
                    overlapping[i] = int(overlapping[i])
            else:
                overlapping = int(overlapping)
        
        if overlapping is not None:
            if type(overlapping) == int: 
                overlapping = [overlapping]
            for o in overlapping:
                all_overlapping.append(o)
                all_overlapping = add_all_contours(all_overlapping,o)
        return all_overlapping

    object_numb = 0
    for i in df_rois.index:
        if df_rois.loc[i,"object_id"] == None:
            df_rois.loc[i,"object_id"] = object_numb
            all_overlapping = add_all_contours([],i)
            for o in all_overlapping: 
                df_rois.loc[o,"object_id"] = object_numb

            object_numb = object_numb + 1
    
    objects["object_id"] = df_rois["object_id"].unique()
    objects["at_edge"] = False 
    for i in objects.index:
        this_object = df_rois[df_rois["object_id"]==objects.loc[i,"object_id"]]
        x_res = this_object["x_res_um"].iloc[0]
        y_res = this_object["y_res_um"].iloc[0]
        z_res = this_object["z_res_um"].iloc[0]


        volume = 0
        contours_ids = None
        contours_centers_xyz = None
        sum_grey_channels = [0]*len(channels)
        is_inside_channels = ["tempstring"]*len(channels)
        at_edge = False 
        for j in this_object.index: 
            if this_object.loc[j,"at_edge"]:
                at_edge = True
            
            volume = volume + (this_object.loc[j,"area"]*x_res*y_res*z_res)
            
            if contours_ids is None: 
                contours_ids = str(j)
            else: 
                contours_ids = contours_ids + "_" +str(j)
            
            if contours_centers_xyz is None: 
                contours_centers_xyz = "x" + str(this_object.loc[j,"centroid_x"]) +"y" +str(this_object.loc[j,"centroid_y"])+"z"+str(this_object.loc[j,"z_int"])
            else: 
                contours_centers_xyz = contours_centers_xyz + "_x" + str(this_object.loc[j,"centroid_x"]) +"y" +str(this_object.loc[j,"centroid_y"])+"z"+str(this_object.loc[j,"z_int"])
            
            for c in range(len(channels)): 
                sum_grey_channels[c] = sum_grey_channels[c] + this_object.loc[j,"sum_grey_"+channels[c]]*x_res*y_res*z_res 
                is_inside_channels[c] = is_inside_channels[c] +","+ str(this_object.loc[j,"is_inside_"+channels[c]])
                is_inside_channels[c] = is_inside_channels[c].replace("tempstring,","")
                is_inside_channels[c] = is_inside_channels[c].replace("tempstring","")
            

        objects.loc[i,"root"] = this_object["root"].iloc[0]
        objects.loc[i,"file"] = this_object["file"].iloc[0]
        objects.loc[i,"z_stack_id"] = this_object["z_stack_id"].iloc[0]
        
        objects.loc[i,"volume"] = volume
        objects.loc[i,"contour_ids"] = contours_ids 
        objects.loc[i,"contours_centers_xyz"] = contours_centers_xyz 
        objects.loc[i,"at_edge"] = at_edge
        
        for c in range(len(channels)):
            objects.loc[i,"sum_grey_"+channels[c]] = sum_grey_channels[c]
            objects.loc[i,"is_inside_"+channels[c]] = is_inside_channels[c]

    return objects
    
def make_many_projections(image_paths,temp_folder,projections_folder,channel_colors,cores=None,auto_max=True):
    '''
    Take in multiple paths to images in microscopy format such as .lsm or .czi and make minimal projections out of all of them in parallell
    
    Params
    image_paths        : list of str  : Paths to images to make projections of
    temp_folder        : str          : Path to temp folder for storing output
    projections_folder : str          : Folder to store output projections in
    cores              : int          : cores to use for parallel processing
    channel_colors     : tuple of int : Color to convert black and white projection into
    auto_max           : bool         : Readjust level to the maximum after making projection
    
    Returns
    
    '''

    if cores is None:
        cores = mp.cpu_count()-1

    tot_images = len(image_paths)
    image_info = []
    if cores == 1: 
        for i in range(0,len(image_paths)):
            info = str(i+1)+"/"+str(tot_images)
            image_info.append(make_projections(image_paths[i],info,temp_folder,projections_folder,channel_colors,auto_max))
    else:
        pool = mp.Pool(cores)

        for i in range(0,len(image_paths)):
            info = str(i+1)+"/"+str(tot_images)
            image_info.append(pool.apply_async(make_projections,args=(image_paths[i],info,temp_folder,projections_folder,channel_colors,auto_max)))

        pool.close()
        pool.join()

        image_info = [x.get() for x in image_info]
        
    return None
        
def make_projections(image_id,info,temp_folder,projections_folder,channel_colors,auto_max):
    '''
    Take in path to one image in microscopy format such as .lsm or .czi and make colored maximal projections. One projection per channel and a composite image.

    Params
    image_id           : str          : Path to image to microscopy image file to process
    info               : str          : Info string about image that is being processed
    temp_folder        : str          : Path to temp folder for storing output
    projections_folder : str          : Folder to store output projections in
    cores              : int          : cores to use for parallel processing
    channel_colors     : tuple of int : Color to convert black and white projection into
    auto_max           : bool         : Readjust level to the maximum after making projection
    '''

    img_name = os.path.splitext(os.path.split(image_id)[1])[0]
    
    images_raw_extracted_folder = os.path.join(temp_folder,img_name)
    images_raw_extracted_infofile = os.path.join(images_raw_extracted_folder,"files_info.csv")
    if os.path.isfile(images_raw_extracted_infofile):
        print(info+"\tImage_id: "+image_id+"\tImages already extracted, using those.")
        images_paths = pd.read_csv(images_raw_extracted_infofile,index_col = 0)
    else: 
        print(info+"\tImage_id: "+image_id+"\tExtracting images...")
        images_paths = get_images(image_id,images_raw_extracted_folder)

    for t in images_paths["T_index"].unique():
        this_time = images_paths.loc[images_paths["T_index"]==t].copy()
        
        for s in this_time["series_index"].unique():
            this_series = this_time.loc[this_time["series_index"]==s].copy()
            all_projections = []
            
            for c in this_series["channel_index"].unique():
                all_z_planes = this_series.loc[this_series["channel_index"]==c].copy()
                
                all_z_planes_paths = []
                for i in all_z_planes.index:
                    all_z_planes_paths.append(os.path.join(all_z_planes.loc[i,"root"],all_z_planes.loc[i,"file"]))

                projection = make_single_projection(all_z_planes_paths,max_projection = True,auto_max = auto_max)
                
                channel_i = int(c.replace("C",""))
                projection = gray_to_color(projection,channel_colors[channel_i])
                
                img_dim = str(projection.shape[0])+"x"+str(projection.shape[1])
                
                if auto_max: 
                    auto_max_string = "autoleveled"
                else: 
                    auto_max_string = "notleveled"
                
                projection_path = os.path.join(projections_folder,img_name+"_INFO_"+img_dim+"_"+str(s)+"_"+str(t)+"_"+str(c)+"_"+auto_max_string+".png")
                cv2.imwrite(projection_path,projection)

                all_projections.append(projection)

            composite = None
            for p in all_projections:
                if composite is None:
                    composite = p
                else: 
                    composite = cv2.add(composite,p)
            
            img_dim = str(projection.shape[0])+"x"+str(projection.shape[1])
            composite_path = os.path.join(projections_folder,img_name+"_INFO_"+img_dim+"_"+str(s)+"_"+str(t)+"_Ccomp.png")
            cv2.imwrite(composite_path,composite)
    
    return None

def plot_images_pdf(save_folder,df,file_vars,x_vars,y_vars,image_vars,image_dim,sort_ascending=None):
    '''
    Create PDFs with images grouped after variables
    
    Params
    df              : pd.Dataframe : Data frame with file paths to images (df["file_path"]) and their variables to sort them by. rows = images, cols = variables
    file_vars       : list of str  : Column names of varibles used to make different PDF files
    x_vars          : list of str  : Column names of variables to plot on x-axis
    y_vars          : list of str  : Column names of variables to plot on y-axis
    image_vars      : list of str  : Column names of variables to print on top of image
    image_dim       : tuple of int : image dimensions
    sort_ascending : list of bool  : Directions to sort each column in, file_vars then y_vars then x_vars. Must be equal in length to 1+len(x_vars)+len(y_vars) (file_vars is compressed to 1 column). True = ascending sort, False = descending sort.  

    Returns
    df_plot      : pd.DataFrame    : Data frame describing the location of images in the pdf
    '''
    
    os.makedirs(save_folder,exist_ok=True)

    df["file_vars"] = ""
    sep=""
    for i in file_vars: 
        temp = df[i].astype(str).str.replace("_","")
        df["file_vars"] = df["file_vars"] + sep + temp
        sep="_"
    
    df["x_vars"] = ""
    for i in x_vars: 
        temp = df[i].astype(str).str.replace("_","")
        df["x_vars"] = df["x_vars"] +"_"+ temp
    
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
        make_pdf(save_path,df_f,x_vars,y_vars,image_vars,image_dim)
    
    return "Made pdf!"

def make_pdf(save_path,df,x_vars,y_vars,image_vars,image_dim): 
    '''
    Input df with images and make a pdf out of them

    Params
    save_path    : str          : Path to save pdf in
    df           : pd.DataFrame : Data frame with paths to images and columns to sort images by
    x_vars       : list of str  : Column names of variables to plot on x-axis
    y_vars       : list of str  : Column names of variables to plot on y-axis
    image_vars   : list of str  : Column names of variables to print on top of image 
    image_dim    : tuple of int : image format in pixels [width,height]. If not all images have this aspect ratio, some might end up overlapping. 
    
    '''
    
    def shorten_info(info): 
        short_info = info
        if len(short_info)>22:
            short_info = short_info.split(" = ")[1]
            if len(short_info)>22:
                short_info = short_info[0:22]
        #print("info: "+info+"\tlen: "+str(len(info))+"\tshorten_info: "+short_info+"\tlen: "+str(len(short_info)))
        return short_info
    
    canvas = Canvas(save_path,pagesize = A4)

    c_width,c_height = A4

    area_meta_yvars = (0,len(y_vars)*15)
    area_meta_xvars = (0,len(x_vars)*15)
    marg = 5
    goal_aspect_ratio = image_dim[1]/image_dim[0]

    imgs_per_col = df["pdf_x_position"].max()+1
    img_width = (c_width-area_meta_yvars[1])/imgs_per_col
    img_height = img_width * goal_aspect_ratio

    if img_height > c_height: 
        img_height = c_height
    
    y_img_per_page = int((c_height-area_meta_yvars[1])/img_height)
    
    yvars_box_width  = (area_meta_yvars[1]-area_meta_yvars[0])/(len(y_vars))
    xvars_box_height = (area_meta_xvars[1]-area_meta_yvars[0])/len(x_vars)
     
    page = 0
    for i in df.index:
        image_path = df.loc[i,"file_path"]
        image = ImageReader(image_path)

        x = (df.loc[i,"pdf_x_position"]*img_width) + area_meta_yvars[1]
        y_position = df.loc[i,"pdf_y_position"] - (page*y_img_per_page) + 1
        if y_position > y_img_per_page: 
            page = page +1 
            canvas.showPage()
            y_position = y_position- y_img_per_page
        y = c_height - ((y_position)*img_height)- area_meta_xvars[1]
        
        image_x = x+marg/2
        image_y = y+marg/2
        canvas.drawImage(image,image_x,image_y,width = img_width-marg, height = img_height-marg)
        
        image_info = "\t"
        for j in image_vars: 
            image_info = image_info +str(df.loc[i,j]) + "   "
        image_info = image_info.replace("\t","",1)
        
        canvas.setFont("Helvetica",4)
        canvas.setFillColorRGB(0.5,0.5,0.5)
        canvas.drawString(image_x+1,image_y+1,image_info)       
        
        canvas.saveState()
        
        last_info = ""
        for j in range(len(y_vars)): 
            info = y_vars[j]+" = "+str(df.loc[i,y_vars[j]])
            if info != last_info:
                x_ymeta = yvars_box_width*j  
                draw_annotation_pdf(canvas,[x_ymeta+marg/4,y+marg/2,yvars_box_width-marg/2,img_height-marg],True,shorten_info(info))
                last_info = info
        
        last_info = ""
        for j in range(len(x_vars)): 
            info = x_vars[j]+" = "+str(df.loc[i,x_vars[j]])
                
            if info != last_info:
                y_xmeta = xvars_box_height*(j+1) 
                draw_annotation_pdf(canvas,[x+marg/2,c_height-(y_xmeta+marg/4),img_width-marg,xvars_box_height-marg/4],False,shorten_info(info))
        canvas.restoreState()
    canvas.save()

    print("Saved pdf: "+save_path)

def draw_annotation_pdf(canvas,box_dims,vertical_text,text):
    '''
    Make an annotation box in your pdf document. NB! Remember to set font and colors before invoking function. 

    Params
    canvas     : reportlab.Canvas : pdf dokument to draw box in
    box_dims   : tuple of float   : Position of box (x1,y1,width,height)
    text_angle : float            : Angle to rotate text by 
    '''
    from reportlab.lib.colors import Color
    
    canvas.setFont("Helvetica",8)
    canvas.setStrokeColorRGB(1,1,1)
    canvas.setFillColor(Color(0.8,0.8,0.8,alpha=0.8))
    canvas.rect(box_dims[0],box_dims[1],box_dims[2],box_dims[3],fill=1,stroke=0)
    canvas.setFillColor(Color(0,0,0,alpha=1))
    
    text_nudge = 3
         
    if vertical_text: 
        x_text = box_dims[1]+(box_dims[3]/2) 
        y_text = box_dims[0]+(box_dims[2]/2+text_nudge) 
        canvas.saveState()
        canvas.rotate(90)
        canvas.drawCentredString(x_text,-y_text,text)
        canvas.restoreState()
    else: 
        x_text = box_dims[0]+(box_dims[2]/2) 
        y_text = box_dims[1]+(box_dims[3]/2-text_nudge) 
        canvas.drawCentredString(x_text,y_text,text)

def imcrop2(img,bbox,value=255):
    """
    Crop image with black border if it is outside image
    
    Params
    img     : numpy array   : image
    bbox  	: list          : [x1,y1,x2,y2]
    """
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2,value=value)
    return img[y1:y2, x1:x2]

def imcrop(img,center,width,value=255):
    """
    Crop image with black border if it is outside image
    
    Params
    img     : numpy array   : image
    center  : list          : [x,y]
    width   : int           : width of image
    """
    x1 = int(center[0] - width/2)
    y1 = int(center[1] - width/2)
    x2 = int(center[0] + width/2)
    y2 = int(center[1] + width/2)
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

def remove_small_objects(image,min_size):
    """
    Remove all white objects in an bw image that is below min_size

    Params
    image     : numpy array  : 8 bit single channel
    min_size  : int          : minimal size of object

    Returns
    img2      : numpy array  : 8 bit single channel

    """    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def euc_distance(x1,y1,x2,y2):
    """
    Eucleudian distance between (x1,y1) and (x2,y2)
    """
    return(math.sqrt((x2-x1)**2 + (y2-y1)**2))

def find_edges(image_path,canny_threshold = (50,125),smallest_object = 20):
    """
    Find edge of obects in an image and return the mask of those edges

    Params
    image_path         : str        :  A path to an image to find edges of
    canny_threshold    : int tupple :  Lower and upper bound of canny edge detection
    smallest_object    : int        :  Objects smaller than this are considered noise and filtered out

    Returns
    edges : numpy.array : The minimal projection of found edges

    """
    edge = cv2.imread(image_path,cv2.IMREAD_ANYDEPTH)
    max_pixel = np.max(np.amax(edge,axis=0))
    edge = (edge/(max_pixel/255)).astype('uint8')
    edge = cv2.Canny(edge,canny_threshold[0],canny_threshold[1])
    edge = remove_small_objects(edge,smallest_object)
    
    return edge

def make_single_projection(z_planes,max_projection = True,auto_max=True):
    '''
    Make projections from z_planes by finding the minimal or maximal value in the z-axis

    Params 
    z_planes        : list : a list of paths to each plane in a stack of z_planes
    max_projection  : bool : Use maximal projection, false gives minimal projection
    auto_max        : bool : Auto level image after making projection

    Returns
    projection: numpy.array : the projection of the z_planes as an 8-bit image 

    '''
    projection = None
    for path in z_planes: 
        plane = cv2.imread(path,cv2.IMREAD_ANYDEPTH)
        if projection is None:
            projection = plane
        else:
            if max_projection: 
                projection = np.maximum(projection,plane)
            else: 
                projection = np.minimum(projection,plane)
           
    max_pixel = projection.max()
    projection = (projection/(max_pixel/255)).astype('uint8')
    
    return projection

def gray_to_color(gray_img,bgr): 
    '''
    Convert grayscale image to rgb image in the color specified

    Params
    gray_img : numpy.array  : grayscale cv2 image with values 0 -> 255
    bgr      : int tupple : bgr color (b,g,r) where 0 <= b,g,r <= 255

    Returns
    color   : np.array    : RGB image
    '''
    color = np.zeros((gray_img.shape[0],gray_img.shape[1],3),np.uint8)
    color[:] = bgr
    gray_img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR)
    gray_img = gray_img.astype('float') / 255
    out = np.uint8(gray_img*color)
    
    return out

def extract_contour(image,contour,center,width):
    '''
    Take in an image and a contour and draw the pixels in the image that are enclosed by the contour with a white background

    Params
    image   : numpy.array   : image with pixel values
    contour : list          : a cv2 type contour
    center  : list          : [x,y]
    width   : int           : width of image
    name    : str           : output name of contour
    
    Return
    image   : numpy.array   : only contour
    '''

    mask = np.zeros(image.shape,np.uint8)
    cv2.drawContours(mask,[contour],0,255,thickness=cv2.FILLED)
    
    mask  = imcrop(mask,center,width,0)
    image = imcrop(image,center,width,0)
    
    image = cv2.bitwise_not(image)
    image = cv2.bitwise_or(image,image,mask=mask)
    image = cv2.bitwise_not(image)
    return image

def watershed(contours_full,contours_centers):	
    '''
    Takes in one bw image specifying the centers of contours and one specifying edges. Then uses watershed to flood from centers to edges and thus split contours

    Params
    contours_full    : numpy.array : bw image where 255 specify the full contours to split
    contours_centers : numpy.array : bw image where 255 specify the centers of contours to split from 
    
    Returns
    contours         : list        : wathershedded cv2 contours
    '''
    unknown = cv2.subtract(contours_full,contours_centers)

    ret,markers = cv2.connectedComponents(contours_centers)
    markers = markers + 1
    markers[unknown==255] = 0
    
    contours_full = cv2.cvtColor(contours_full, cv2.COLOR_GRAY2BGR)

    markers = cv2.watershed(contours_full,markers)
    
    #Output of watershed is an image where each pixel is an integer specifying which object it is
    #Here we draw each of these objects find the contour. 0 and 1 is omited as they are background
    img_zero = np.zeros(contours_centers.shape,dtype="uint8")
    contours = []
    for i in range(2,markers.max()+1):
        single_contour = img_zero.copy()
        single_contour[markers==i] = 255
        contour_i, hierarchy = cv2.findContours(single_contour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = contours + contour_i
    
    return contours
    
def find_centers(image,contours):
    '''
    Reveal centers of organoids. More info at cv2s wathershed toutorial: https://docs.opencv.org/3.4.3/d3/db4/tutorial_py_watershed.html

    Params
    image :  numpy array : grayscale image where objects are white and background black

    Returns
    centers : pandas.DataFrame : dataframe with centers
    '''
    #Erode to clean up edges
    kernel = np.ones((3,3),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)

    #distance transform
    image_dist = cv2.distanceTransform(image,cv2.DIST_L2,5)
    ret,image_centers = cv2.threshold(image_dist,10,255,cv2.THRESH_TOZERO)
    image_centers = np.uint8(image_centers)
    
    #Remove noise
    image_centers = remove_small_objects(image_centers,9)

    #check if organoid completely disappeared, if it did add the maximal distance transform center back to image
    img_zero = np.zeros_like(image)
    for i in range(len(contours)): 
        contour_img = img_zero.copy()
        cv2.drawContours(contour_img,contours,i,color=255,thickness = -1)
        extracted = cv2.bitwise_and(image_centers,image_centers,mask=contour_img)
        
        if extracted.max() == 0:
            this_org_dist_transform = cv2.bitwise_and(image_dist,image_dist,mask = contour_img)
            center = this_org_dist_transform.argmax()
            center = np.unravel_index(center,this_org_dist_transform.shape)
            xc = center[1]
            yc = center[0]
            cv2.rectangle(image_centers,(xc-2,yc-2),(xc+2,yc+2),color=255)
    
    #write found center contours to image and find the distance transform center
    contours_center,hierarchy_center = cv2.findContours(image_centers.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_contours_center = img_zero.copy()
    cv2.drawContours(img_contours_center,contours_center,-1,color=255,thickness = -1)
    img_contours_center = cv2.distanceTransform(img_contours_center,cv2.DIST_L2,5)
    
    xc_list = []
    yc_list = []
    for i in range(len(contours_center)):
        contour_img = img_zero.copy()

        cv2.drawContours(contour_img,contours_center,i,color=255,thickness = -1)
        extracted = cv2.bitwise_and(img_contours_center,img_contours_center,mask=contour_img)

        center = extracted.argmax()
        center = np.unravel_index(center,extracted.shape)

        xc_list.append(center[1])
        yc_list.append(center[0])
        
    out = pd.DataFrame(data={'X':xc_list,'Y':yc_list})
    return out

def find_closest_contour(annotation_x,annotation_y,contours): 
    '''
    Find the contour that is closest to the annotated center, i.e. all annotations get a contour, but not necessarily all contours get an annotation 
    
    Params
    annotation_x : list : list of the x-coordinates of the annotated center
    annotation_y : list : list of the y-coordinates of the annotated center
    contours     : list : cv2 formated contours
    Return
    closest_contour : list : For every annotated center returns the index of the closest contour in countours
    '''
    
    x_contours = []
    y_contours = []
    
    for c in contours:
        M = cv2.moments(c)
        x_contours.append(int(M['m10']/M['m00']))
        y_contours.append(int(M['m01']/M['m00']))
    
    closest_org_list = []
    for i in range(len(annotation_x)): 
                
        closest_org = None
        current_distances = []
        for j in range(0,len(contours)):
            d = euc_distance(annotation_x[i],annotation_y[i],x_contours[j],y_contours[j])
            current_distances.append(d)
            if (d < 10):
                closest_org = j
                break

        if closest_org is None: 
            closest_org = current_distances.index(min(current_distances)) 
        closest_org_list.append(closest_org)
        
    return closest_org_list

def find_contour(annotation,contours): 
    '''
    Find the contour that encompass the annotated center
    
    Params
    annotation  : pd.DataFrame  : Data frame with all the annotaiton data. X and Y is column with coordinates for annotated center 
    contours    : list          : cv2 formated contours

    Return
    annotation : pd.DataFrame : Data frame with all the annotation data. Added the column 'closest_organoid' 
    '''
    
    annotation['closest_contour'] = pd.Series(np.NaN,dtype="Int64")
    
    for i in range(0,len(contours)):
        for j in annotation.index:
            if cv2.pointPolygonTest(contours[i],(annotation.loc[j,'X'],annotation.loc[j,'Y']),False) >= 0:
                if(annotation.loc[j,'closest_contour'] is not np.NaN):
                    print("Warning: Annotation center inside two shapes. Picking last one.")
                annotation.loc[j,'closest_contour'] = i 
    
    return annotation
        
def get_shape_info(contours,mask,channels_to_measure,shrink,min_size = 3):
    #TODO: add eccentricity, convex area, euler number, extent, major and minor axis length
    '''
    Get a diverse set of measurements for the countour that is closest to each annotated center

    Params
    contours            : list       : cv2 type contours
    mask                : cv2.image  : The mask that contours was found in 
    channels_to_measure : dictionary : Dictionary of image_paths that should be measured with the contour. Typically each channel of a confocal image. {channel_name:image_path}.
    shrink              : float      : Value to shrink channels so that they match the mask size. Used for performance gains
    min_size            : float      : minimium area, objects of smaller size are filtered out. 

    Returns
    annotaiton      : pandas.DataFrame  : The same annotation data frame as input, just with the added values

    '''
    col_names = ["contour_i","at_edge","area","centroid_x","centroid_y","perimeter","hull_tot_area","hull_defect_area","solidity","radius_enclosing_circle","equivalent_diameter","circularity","contour","img_dim"]
    
    colnames_channels = []
    for key in channels_to_measure:
        colnames_channels.append("sum_grey_"+key)
    
    col_names = col_names + colnames_channels

    df = pd.DataFrame(columns = col_names,index = list(range(0,len(contours))))
    
    for i in df.index:
        df.loc[i,"contour_i"] = i
        df.loc[i,"contour"] = contours[i]
        df.loc[i,"img_dim"] = mask.shape
        c = contours[i]
        M = cv2.moments(c)
        M = cv2.moments(c)
        df.loc[i,'area'] = M['m00']
        if df.loc[i,'area'] > min_size:
            if df.loc[i,'area'] > 0:
                df.loc[i,'centroid_x'] = int(M['m10']/M['m00'])
                df.loc[i,'centroid_y'] = int(M['m01']/M['m00'])
            df.loc[i,'perimeter'] = cv2.arcLength(c,True)
            df.loc[i,'hull_tot_area'] = cv2.contourArea(cv2.convexHull(c))
            df.loc[i,'hull_defect_area'] = df.loc[i,'hull_tot_area'] - df.loc[i,'area']
            if df.loc[i,'hull_tot_area'] > 0:
                df.loc[i,'solidity'] = df.loc[i,'area']/df.loc[i,'hull_tot_area']
            (x,y),radius = cv2.minEnclosingCircle(c)
            df.loc[i,'radius_enclosing_circle'] = radius
            df.loc[i,'equivalent_diameter'] = np.sqrt(4*df.loc[i,'area']/np.pi)
            if df.loc[i,'perimeter'] > 0:
                df.loc[i,'circularity'] = float(4) * np.pi * df.loc[i,'area']/(df.loc[i,'perimeter']*df.loc[i,'perimeter'])
            
            only_contour = np.zeros(mask.shape,np.uint8)
            cv2.drawContours(only_contour,[c],0,255,-1)
            contour_at_edge = (np.sum(only_contour[0,:])> 0) or (np.sum(only_contour[:,0]) > 0)or (np.sum(only_contour[:,mask.shape[0]-1])>0) or (np.sum(only_contour[mask.shape[1]-1,:])>0)
            
            df.loc[i,"at_edge"] = contour_at_edge

            for channel in channels_to_measure:
                channel_path = channels_to_measure[channel]
                if channel_path is not None: 
                    channel_img = cv2.imread(channel_path,cv2.IMREAD_ANYDEPTH) 
                
                    if not (0.999999 < shrink < 1.000001):
                        new_dim = (int(channel_img.shape[0]*shrink),int(channel_img.shape[1]*shrink))
                        channel_img = cv2.resize(channel_img,new_dim,interpolation = cv2.INTER_AREA) 
                
                    if mask.shape != channel_img.shape:
                        raise ValueError("shape channel: "+str(channel_img.shape)+"\t shape mask: "+str(mask.shape)+". Channel and mask shape have to be equal.")
                    only_shape = cv2.bitwise_or(channel_img,channel_img,mask=only_contour)
                    df.loc[i,'sum_grey_'+channel] = only_shape.sum()
        
    df = df[df["area"]>min_size]

    return df

def get_overlapping_contours(this_z,next_z):
    '''
    For each contour in this z-plane find all overlapping contours in next plane

    Params
    this_z    : pd.DataFrame : df with info on contours of this z_plane 
    next_z    : pd.DataFrame : df with info on contours of the next z_plane

    Returns
    overlapping : list of str : list of contour ids that overlap between the two z_planes
    '''
    
    empty_img = np.zeros(this_z["img_dim"].iloc[0],dtype="uint8")
    
    overlapping = []
    for this_i in this_z.index:
        this_z_contour = empty_img.copy()
        cv2.drawContours(this_z_contour,[this_z.loc[this_i,"contour"]],-1,color=255,thickness = -1)
        overlap_this_i = None
        for next_i in next_z.index: 
            next_z_contour = empty_img.copy()
            cv2.drawContours(next_z_contour,[next_z.loc[next_i,"contour"]],-1,color=1,thickness = -1)
            overlap_img = cv2.bitwise_and(next_z_contour,next_z_contour,mask = this_z_contour)
            if overlap_img.sum() > 5:
                if overlap_this_i is not None:
                    overlap_this_i = str(overlap_this_i) +","+str(next_i)
                else: 
                    overlap_this_i = str(next_i)
        overlapping.append(overlap_this_i)
    return overlapping

def check_if_inside(this_z,other_z):
    '''
    check if this contour is at least 60 % covered by another contour in another image and return that contour. 

    Params
    this_z    : pd.Series    : Info on this contour
    other_z   : pd.DataFrame : df with info on contours of the next z_plane

    Returns
    is_inside : list of str  : list of contour ids that is inside this contour
    '''
    
    empty_img = np.zeros(this_z["img_dim"],dtype="uint8")
    
    this_z_contour = empty_img.copy()
    cv2.drawContours(this_z_contour,this_z["contour"],-1,color=255,thickness = -1)
    is_inside = None
    #print("contour_i: "+str(this_z["info"])+" "+str(this_z["contour_i"]),end="\t")
    for other_i in other_z.index: 
        other_z_contour = empty_img.copy()
        cv2.drawContours(other_z_contour,[other_z.loc[other_i,"contour"]],-1,color=1,thickness = -1)
        overlap_img = cv2.bitwise_and(other_z_contour,other_z_contour,mask = this_z_contour)
        if overlap_img.sum() > (other_z_contour.sum()*0.6) :
            if is_inside is None:
                is_inside = other_i
                #print("is_inside: "+other_z.loc[other_i,"channel_index"],end="\t")
            else:
                warnings.warn("Found multiple contours that envelops more than 60 %. That should not be possible")
    #print("") 
    return is_inside 

def draw_classification(image,annotation):
    '''
    Draw the classification in the image

    Params
    image       : cv2 BGR image : Minimal projection of organoids to draw annotation to 
    annotation  : pd.DataFrame  : data frame with annotation data. columns needed: 'X' = x-position, 'Y' = y-position, 'type' = class of contour

    Returns
    image       : cv2 BGR image : Same as in Params just with annotations drawn
    '''

    classes = list(set(annotation['type']))
    classes.sort()
    classes = ["None"]+classes
    classes = pd.DataFrame({'class': classes,'color':[(255,255,255)]*len(classes)})

    classes_colors = [(255,255,255),(0,255,0),(255,0,255),(255,0,0),(0,255,255),(255,255,0)]
    if len(classes_colors) > len(classes['color']):
        classes['color'] = classes_colors[0:len(classes['color'])]
    else: 
        classes.loc[0:len(classes_colors),'color'] = classes_colors 
    
    for i in classes.index:
        x = 20 
        y = 40 + 12 * i
        cv2.putText(image,classes.loc[i,'class'],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,classes.loc[i,'color'],1,cv2.LINE_AA)
    
    for i in annotation.index:
        x = int(annotation.loc[i,"X"])
        y = int(annotation.loc[i,"Y"])
        
        class_i = annotation.loc[i,'type']
        if class_i is None: 
            class_i = "None"
        this_color = classes.loc[classes['class']==class_i,'color'].iloc[0]
        
        cv2.circle(image,(x,y),2,color=this_color,thickness=-1)
        cv2.putText(image,str(annotation.loc[i,"org_id"]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1,cv2.LINE_AA)
    return image


