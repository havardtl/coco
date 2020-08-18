# This file contains classes to process images

import sys 
import os
import math
import numpy as np
import pandas as pd
import cv2
from io import StringIO 
import copy
import datetime
import aicspylibczi
import pathlib
import xml.etree.ElementTree as ET

from coco_package import info
from coco_package import make_pdf

#TODO: Manual corrections should point to a new annotation file, not edit the old one. 
#TODO: all functionality should happen in channel class. Z-stack and Image should be wrappers to add additonal functionality on top of that. 

VERBOSE = False 

#sys.setrecursionlimit(10**6) 

TEMP_FOLDER = "coco_temp"
CONTOURS_STATS_FOLDER = "contour_stats"
GRAPHICAL_SEGMENTATION_FOLDER = "segmented_graphical"

THIS_SCRIPT_FOLDER = os.path.split(os.path.abspath(__file__))[0]

DEFAULT_SETTINGS_XLSX = os.path.join(THIS_SCRIPT_FOLDER,"default_annotations.xlsx")
TEST_SETTINGS_SHEET = "test_settings"
SEGMENT_SETTINGS_SHEET = "segment_settings"
ANNOTATION_SHEET = "annotation"
PLOT_VARS_SHEET = "plot_vars"

contour_id_counter = 0

#####################################
# Functions
#####################################

def set_verbose(): 
    VERBOSE = True
    info.set_verbose()
    make_pdf.set_verbose()

def set_zipped_list(a,b): 
    '''
    Get set of two list that are linked together

    Params
    a : list : list a
    b : list : list b
    '''
    a_set = []
    b_set = []
    for a_i,b_i in set(zip(a,b)): 
        a_set.append(a_i)
        b_set.append(b_i)
    return a_set,b_set
    
###############################################
# Classes
###############################################

class Segment_settings: 

    def __init__(self,channel_index,global_max,color,contrast,auto_max,thresh_type, thresh_upper, thresh_lower,open_kernel,close_kernel,min_size,combine):
        '''
        Object for storing segmentation settings

        Params
        channel_index   : int           : Channel index of channnel these settings account for
        global_max      : int           : Global threshold to apply before making images into 8-bit greyscale
        color           : str           : Color to convert channel into if needed in 8-bit BGR format. "B,G,R" where 0 <= b,g,r <= 255
        contrast        : float         : Increase contrast by this value. Multiplied by image. 
        auto_max        : bool          : Autolevel mask before segmenting
        thresh_type     : str           : Type of threshold to apply. Either ["canny_edge","binary"]
        thresh_upper    : int           : Upper threshold value
        thresh_lower    : int           : Lower threshold value
        open_kernel     : int           : Size of open kernel 
        close_kernel    : int           : Size of closing kernel 
        min_size        : int           : Remove all objects with area less than this value
        combine         : bool          : Whether or not this channel should be combined into combined value 
        '''
        global VERBOSE,TEMP_FOLDER,CONTOURS_STATS_FOLDER 
        
        self.AVAILABLE_THRESH_TYPES = ["canny_edge","binary"] 
        
        self.channel_index = self.to_int_or_none(channel_index)
        self.id_channel = "C"+str(self.channel_index) 
        self.global_max = self.to_int_or_none(global_max)
        
        if color is None: 
            self.color = color
        else: 
            b,g,r = color.split(",",3)
            self.color = (int(b),int(g),int(r))
        
        self.contrast = self.to_float_or_none(contrast)
        self.auto_max = self.to_bool_or_none(auto_max)
        
        assert thresh_type in self.AVAILABLE_THRESH_TYPES,"Chosen thresh type not in AVAILABLE_THRESH_TYPES: "+str(self.AVAILABLE_THRESH_TYPES)
        self.thresh_type = thresh_type
        
        thresh_upper = self.to_int_or_none(thresh_upper)
        thresh_lower = self.to_int_or_none(thresh_lower)
        
        if thresh_upper is not None: 
            assert 0<=thresh_upper<256,"thresh_upper not in range (0,255)"
        
        if thresh_lower is not None: 
            assert 0<=thresh_lower<256,"thresh_lower not in range (0,255)"

        if self.thresh_type == "binary": 
            assert thresh_upper == 255,"Upper threshold must be 255 if binary threshold"

        self.thresh_upper = thresh_upper 
        self.thresh_lower = thresh_lower

        self.open_kernel_int = open_kernel
        self.close_kernel_int = close_kernel
        self.open_kernel = self.make_kernel(open_kernel)
        self.close_kernel = self.make_kernel(close_kernel)
        
        self.min_size = self.to_int_or_none(min_size)

        self.combine = combine 

        if VERBOSE: 
            print("Made Segment_settings object: "+str(self))

    def make_kernel(self,size):
        '''
        Make a square kernel of ones with width and height equal size
        Params 
        size   : int : width and height of kernel 
        
        Returns 
        kernel : np.array : Kernel of ones 
        '''
        size = self.check_str_none(size)
        if size is None: 
            return None
        if pd.isna(size): 
            return None 
        
        size = int(size)
        kernel = np.ones((size,size),np.uint8)
        return kernel
     
    def to_float_or_none(self,value): 
        # If value is not none, turn it into a int
        value = self.check_str_none(value)
        if value is not None: 
            return float(value)
        else: 
            return None
    
    def to_int_or_none(self,value): 
        # If value is not none, turn it into a int
        value = self.check_str_none(value)
        if value is not None: 
            return int(float(value))
        else: 
            return None
    
    def to_bool_or_none(self,value): 
        # If value is not none, turn it into a bool
        value = self.check_str_none(value)
        if value is not None: 
            return bool(value)
        else: 
            return None
            
    def check_str_none(self,value):
        #check if value is a None string and in that case make it None
        value_str = str(value)
        value = value_str.strip()
        if value == "None" or value == "none":
            return None
        else:
            return value
    
    def get_dict(self):
        data = {
            "id_channel": self.id_channel,
            "channel_index":self.channel_index,
            "color":self.color,
            "contrast":self.contrast,
            "auto_max":self.auto_max,
            "thresh_type":self.thresh_type,
            "thresh_upper":self.thresh_upper,
            "thresh_lower":self.thresh_lower,
            "close_kernel":self.close_kernel_int,
            "open_kernel":self.open_kernel_int
        }
        return data
    
    @classmethod
    def excel_to_segment_settings(self,excel_path): 
        '''
        Convert Excel file to list of Segment_settings objects
        
        Params 
        excel_path : str : Path to excel file to read
        
        Returns
        segment_settings : list of Segment_settings : List of Segment_settings
        '''
        
        df_segment_settings = pd.read_excel(excel_path,sheet_name = SEGMENT_SETTINGS_SHEET)
        
        df_segment_settings = df_segment_settings.where((pd.notnull(df_segment_settings)), None) #Convert invalid cells to None instead of NaN

        segment_settings = []
        for channel in df_segment_settings["channel_index"]: 
            s = df_segment_settings.loc[df_segment_settings["channel_index"]==channel,]
            if len(s["channel_index"])!=1:
                raise ValueError("Need at least one unique channel index in segment settings.")
            s = s.iloc[0,]
            segment_settings_c = Segment_settings(channel,s["global_max"],s["color"],s["contrast"],s["auto_max"],s["thresh_type"],s["thresh_upper"],s["thresh_lower"],s["open_kernel"],s["close_kernel"],s["min_size"],s["combine"])
            segment_settings.append(segment_settings_c)
            
        return segment_settings

    def __repr__(self):
        string = "{class_str}: channel = {ch}, global_max = {g}, color = {col},contrast = {c},auto_max = {a},thresh_type = {tt},thresh_upper = {tu}, thresh_lower = {tl},open_kernel = {ok}, close_kernel = {ck}\n".format(class_str = self.__class__.__name__,ch = self.channel_index,g=self.global_max,col=self.color,c=self.contrast,a=self.auto_max,tt=self.thresh_type,tu=self.thresh_upper,tl=self.thresh_lower,ok = self.open_kernel_int,ck = self.close_kernel_int)
        return string

class Mask: 

    def __init__(self,mask_path):
        '''
        Mask that can be added to image so that only this part of the z_stack is processed. 

        Params
        mask_path : str : Path to mask. Must have naming convention '{file_id}_*_MASK_{mask_name}.*'
        '''

        self.mask_path = mask_path 
        
        self.file_id = os.path.splitext(os.path.split(self.mask_path)[1])[0]

        self.mask_name = self.file_id.split("_MASK_",1)[1]

        self.mask = cv2.imread(self.mask_path,cv2.IMREAD_GRAYSCALE)
        self.mask_shape = self.mask.shape
        
        self.process_mask()
    
    def process_mask(self): 
        #Make sure mask has smooth edges
        ret,img = cv2.threshold(self.mask,5,255,cv2.THRESH_BINARY)
        kernel = np.ones((20,20),np.uint8) 
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        self.mask = img

    @classmethod 
    def get_mask_list(self,mask_folder): 
        '''
        Convert a path to folder with masks into a list of Mask objects 

        Params
        mask_folder : str : Path to masks 
        '''
        masks = []
        if mask_folder is None: 
            return masks 
        
        mask_paths = os.listdir(mask_folder)
        
        if VERBOSE: print("Found "+str(len(mask_paths))+" masks in: "+mask_folder)
        
        for m in mask_paths: 
            full_path = os.path.join(mask_folder,m)
            masks.append(Mask(full_path))
        
        return masks

    def __repr__(self): 
        return "{class_str}({mask_path}): shape = {s}".format(class_str = self.__class__.__name__,mask_path = self.mask_path,s = self.mask_shape)

class Zstack: 

    def __init__(self,images,file_id,series_index,time_index):
        '''
        Object representing a z_stack
        
        Params
        images            : list : List of Images objects
        file_id           : str  : Id that identifies the file 
        series_index      : str  : Index of image series when imaging multiple location in one go 
        time_index        : str  : Index of time index in timed experiments
        '''
        
        global VERBOSE,TEMP_FOLDER,CONTOURS_STATS_FOLDER
        
        assert type(images) is list,"images must be a list of image objects"
        for i in images: 
            assert isinstance(i,Image),"found an image in images that is not an image object"

        self.images = images
        self.images.sort()
        
        self.file_id = file_id 
        self.series_index = series_index 
        self.time_index = time_index 
        
        self.id_z_stack = str(self.file_id)+"_S"+str(self.series_index)+"_T"+str(self.time_index)
        
        self.mask_save_folder = os.path.join(TEMP_FOLDER,"masks",self.id_z_stack)
        os.makedirs(self.mask_save_folder,exist_ok = True)
        self.xml_folder = os.path.join(TEMP_FOLDER,"xml_files",self.id_z_stack)
        os.makedirs(self.xml_folder,exist_ok = True)
        self.combined_mask_folder = os.path.join(TEMP_FOLDER,"combined_masks",self.id_z_stack)
        os.makedirs(self.combined_mask_folder,exist_ok=True)
        self.img_w_contour_folder = os.path.join(TEMP_FOLDER,"img_w_contour",self.id_z_stack)
        os.makedirs(self.img_w_contour_folder,exist_ok=True)
        self.img_for_viewing_folder = os.path.join(TEMP_FOLDER,"img_for_viewing",self.id_z_stack)
        os.makedirs(self.img_for_viewing_folder,exist_ok=True)

        self.img_dim = self.images[0].get_img_dim()
        self.physical_size = None

        self.projections = None
        self.composite = None

        self.made_combined_masks = False

        self.filter_mask = None
    
    def filter_w_mask(self,masks,allow_no_mask = False): 
        '''
        Split up z_stack into one z_stack per mask
        
        Params
        masks         : list of Mask : Masks to look through and split on
        allow_no_mask : bool         : If false, an error is raised if no masks where found to match z_stack 
        '''
        matching_masks = []
        for m in masks: 
            if self.file_id in m.file_id: 
                if not self.img_dim == m.mask_shape: 
                    raise RuntimeError("Found a mask with matching file id but not matching shape. "+str(m)+" z_stack shape = "+self.img_dim)
                matching_masks.append(m)

        if len(matching_masks) == 0:
            if not allow_no_mask: 
                raise RuntimeError("Did not find any matching masks. len(masks) = "+str(len(masks))+", z_stack: "+str(self))
            else:
                return [self] 
        
        if VERBOSE: print("Found "+str(len(matching_masks))+" filter masks for this z_stack")
        
        new_z_stacks = []
        for m in matching_masks:
            new_z_stack = copy.deepcopy(self)
            for i in new_z_stack.images:
                for j in i.channels:
                    j.filter_w_mask(m)
            
            new_z_stack.filter_mask = m
            new_z_stack.id_z_stack = new_z_stack.id_z_stack + "_" + new_z_stack.filter_mask.mask_name
            print("\t new_z_stack: "+str(new_z_stack))
            new_z_stacks.append(new_z_stack)
        return new_z_stacks 

    def add_annotations(self,annotations):
        '''
        Add annotations to channels

        Params
        annotations : list of Annotation : Annotation objects to check if can be added to z_stack 
        '''
        this_z_stack = []
        for a in annotations: 
            if self.file_id in a.file_id: 
                this_z_stack.append(a)
        if VERBOSE: print("Annotations found for this z_stack: "+str(len(this_z_stack)))

        for i in self.images: 
            for j in i.channels + [i.combined_mask]:
                print("adding annotations to: "+str(j))
                j.add_annotation(this_z_stack)
    
    def split_on_annotations(self): 
        '''
        Split up combined mask based on annotation
        '''
        for i in self.images: 
            i.combined_mask.split_on_annotations()

    def check_annotations(self):
        '''
        Check whether contour objects have annotational objects inside

        Params
        annotation_paths : list of str : Path to annotation files to add to channels 
        '''
        if VERBOSE: print("Checking if contours have annotations inside in "+str(self))
        for i in self.images: 
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            for j in i.channels: 
                j.check_annotation()
            i.combined_mask.check_annotation(other_channel = True)

        if VERBOSE: print("")

    def make_masks(self,save_masks = False):
        '''
        Generate masks in all Images objects
        
        Params
        save_masks : bool : weather or not to save masks in temp folder
        '''
        if VERBOSE: print("Making masks for all images in "+str(self))
        
        mask_path = None
        if save_masks: 
            mask_path = self.mask_save_folder

        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            i.make_masks(mask_path)
            
        if VERBOSE: print("")
        
    def make_combined_masks(self,use_filter_mask = False):
        '''
        Combine all masks into one to make a super object that other contours can be checked if they are inside
        
        Params
        use_filter_mask : bool : Instead of making combined channel from combining masks, use the filter mask as a combined mask
        '''
        if use_filter_mask: 
            if VERBOSE: print("Setting filter masks as combined_masks for all images in "+str(self))
            
            for i in self.images:
                if VERBOSE: print(i.z_index,end="  ",flush=True)
                i.set_combined_channel(self.id_z_stack,self.combined_mask_folder,self.filter_mask.mask,self.filter_mask.mask_path)
            if VERBOSE: print("")
        
        else: 
            if VERBOSE: print("Making combined channels for all images in "+str(self))

            for i in self.images:
                if VERBOSE: print(i.z_index,end="  ",flush=True)
                i.make_combined_channel(self.id_z_stack,self.combined_mask_folder)
            
            if VERBOSE: print("")
        self.made_combined_masks = True 

    def find_contours(self,min_contour_area=5): 
        #Find contours in all Images objects
        if VERBOSE: print("Finding contours all images in "+str(self))
        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            all_channels = i.channels 
            if self.made_combined_masks: 
                all_channels = all_channels + [i.combined_mask]
            
            for j in all_channels: 
                j.find_contours(min_contour_area)
        if VERBOSE: print("")
    
    def group_contours(self,pixels_per_group = 250*250): 
        #Group all contours to location groups in all Images objects 
        n_div = int((self.img_dim[0]*self.img_dim[1])/pixels_per_group)
        if VERBOSE: print("Grouping contours in z_stack: "+str(self)+" n_div = "+str(n_div))

        for i in self.images: 
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            for j in i.channels: 
                j.group_contours(n_div)
            
            if self.made_combined_masks: 
                i.combined_mask.group_contours(n_div)
        if VERBOSE: print("")

    def find_z_overlapping(self): 
        #Find all overlapping contours for all contours in images
        if VERBOSE: print("Finding z_overlapping for all images in "+str(self))
        for i in range(len(self.images)-1):
            if VERBOSE: print(i,end="  ",flush=True)
            self.images[i].find_z_overlapping(self.images[i+1])
        
        last_image = self.images[len(self.images)-1]
        last_channels = last_image.channels
        if self.made_combined_masks: 
            last_channels = last_channels + [last_image.combined_mask]
        for i in last_channels: 
            for j in i.contours: 
                j.next_z_overlapping = []
                
        if VERBOSE: print("")
            
    def update_contour_stats(self):
        if VERBOSE: print("Updating contour stats for all contours in "+str(self))
        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            all_channels = i.channels
            if self.made_combined_masks:
                all_channels = all_channels + [i.combined_mask]
            for j in all_channels:
                j.update_contour_stats()
                    
        if VERBOSE: print("")
    
    def find_physical_res(self):
        #Get physical resolution for the z_stack 
        self.physical_size = self.images[0].get_physical_res(self.xml_folder)
        for i in self.images: 
            for j in i.channels: 
                j.x_res = self.physical_size["x"]
                j.x_unit = self.physical_size["unit"]
    
    def set_physical_res(self,physical_size): 
        #Set physical resolution for the z_stack
        self.physical_size = physical_size
        for i in self.images: 
            for j in i.channels: 
                j.x_res = self.physical_size["x"]
                j.x_unit = self.physical_size["unit"]
    
    def is_inside_combined(self):
        #Check whether the contour is inside for all images
        if VERBOSE: print("Finding if all contours are inside combined mask for all images in "+str(self))
        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            i.is_inside_combined() 
        if VERBOSE: print("")
        
    def measure_channels(self):
        #Measure contours for all channels
        if VERBOSE: print("Measuring contour intensity for all channels in "+str(self))
        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            all_channels = i.channels
            
            if self.made_combined_masks: 
                all_channels = i.channels + [i.combined_mask]
            
            for j in all_channels: 
                for k in j.contours:
                    k.measure_channel(i.channels)
        if VERBOSE: print("")

    def write_contour_info(self):
        #Write out all information about contours
        if VERBOSE: print("Writing info about contours ... ",end="",flush=True)
        df = pd.DataFrame() 
        for i in self.images: 
            z_index = i.z_index
            all_channels = i.channels
            if self.made_combined_masks: 
                all_channels = all_channels + [i.combined_mask]
            for j in all_channels:
                temp = j.get_contour_stats()
                df = pd.concat([df,temp],ignore_index = True,sort=False)

        df["z_stack_id"] = self.id_z_stack 
        df["x_res"] = self.physical_size["x"]
        df["y_res"] = self.physical_size["y"]
        df["z_res"] = self.physical_size["z"]
        df["res_unit"] = self.physical_size["unit"] 
        df["img_dim_y"] = self.img_dim[0]
        df["img_dim_x"] = self.img_dim[1]
        df["file_id"] = self.file_id 
        df["series_index"]  = self.series_index  
        df["time_index"] = self.time_index
        
        filter_mask = None
        if self.filter_mask is not None: 
            filter_mask = self.filter_mask.mask_name 
        df["filter_mask"] = filter_mask

        contours_stats_path = os.path.join(CONTOURS_STATS_FOLDER,self.id_z_stack+".csv")
        if VERBOSE: print("Wrote contours info to "+contours_stats_path)
        df.to_csv(contours_stats_path,sep=";",index=False)
    
    def make_projections(self,max_projection = True,auto_max=True,colorize = True,add_scale_bar = True,save_folder = None,grayscale = False):
        '''
        Make projections from z_planes by finding the minimal or maximal value in the z-axis
        
        Params 
        z_planes        : list : a list of paths to each plane in a stack of z_planes
        max_projection  : bool : Use maximal projection, false gives minimal projection
        auto_max        : bool : Auto level image after making projection
        colorize        : bool : Add color to image from segment_settings
        add_scale_bar   : bool : Add scale bar to image
        save_folder     : str  : folder to save projections in. If None, this is stored in temp_folder/raw_projections 
        grayscale       : bool : if True, keep images in grayscale and do not change them to three channel color images 

        Updates
        self.projections : list of np.array : List of the projections of the z_planes as an 8-bit image. Ordered by channel_index.
        self.composite   : np.array         : Composite image of the projections
        '''
        if VERBOSE: print("Making projections for z_stack: "+self.id_z_stack)
        if save_folder is None: 
            save_folder = os.path.join(TEMP_FOLDER,"raw_projections")
        os.makedirs(save_folder,exist_ok=True)

        projections = []
        
        for i in self.images: 
            for j in i.channels:
                make_new_channel = True
                for p in range(len(projections)): 
                    if projections[p].channel_index == j.channel_index: 
                        make_new_channel = False 
                        if max_projection: 
                            projections[p].image = np.maximum(projections[p].get_image(),j.get_image())
                        else:
                            projections[p].image = np.minimum(projections[p].get_image(),j.get_image())
                
                if make_new_channel:
                    new_channel = Channel(j.full_path,j.channel_index,0,j.color,j.categories)
                    file_id = self.id_z_stack + "_" + str(j.id_channel)
                    new_channel_path = os.path.join(save_folder,file_id+".png")
                    new_channel.full_path = new_channel_path 
                    new_channel.file_id = file_id 
                    new_channel.image = j.get_image().copy()
                    projections.append(new_channel)
        
        max_channel_index = 0
        for p in projections: 
            p.x_res = self.physical_size["x"]
            p.x_unit = self.physical_size["unit"]
            p.image = p.get_img_for_viewing(scale_bar=add_scale_bar,auto_max=auto_max,colorize=colorize,grayscale = grayscale)
            if not cv2.imwrite(p.full_path,p.image):
                raise ValueError("Could not save image to: "+p.full_path)
        
            if p.channel_index > max_channel_index: 
                max_channel_index = p.channel_index
        
        self.projections = projections
        
        composite_img = None
        for p in projections: 
            if composite_img is None: 
                composite_img = p.get_image()
            else: 
                composite_img = cv2.add(composite_img,p.get_image()) 
        
        new_channel_path = os.path.join(save_folder,self.id_z_stack+"_Ccomb.png")
        if not cv2.imwrite(new_channel_path,composite_img):
            raise ValueError("Could not save image to: "+new_channel_path)
        composite = Channel(new_channel_path,max_channel_index + 1,0,(255,255,255),j.categories)
        composite.id_channel = "Ccomb"
        composite.image = composite_img
                
        self.composite = composite
        
    def get_projections_data(self): 
        '''
        Get information about projections such as their path and other info

        Returns
        df : pd.DataFrame : Info in DataFrame 
        '''
        full_paths = []
        file_id = []
        channel_index = []
        for p in self.projections+[self.composite]: 
            full_paths.append(p.full_path)
            file_id.append(p.file_id)
            channel_index.append(p.channel_index)

        df = pd.DataFrame({"full_path":full_paths,"file_name":file_id,"channel_index":channel_index})
        
        df["file_id"] = self.file_id
        df["id_z_stack"] = self.id_z_stack 
        df["series_index"] = self.series_index
        df["time_index"] = self.time_index

        df["img_dim"] = str(self.img_dim) 
        
        return df
    
    def get_max_projection_zstack(self,segment_settings):
        '''
        Get the max projection of the z_stack and create a z_stack from that which can be used to process image in 2D instead of 3D. 

        Params
        segment_settings : list of Segment_settings : How Images shall be processed
        
        Returns 
        zstack : Zstack : The same Zstack as this one, but compressed in the z-axis 
        '''
        self.make_projections(max_projection = True,auto_max=False,colorize = False,add_scale_bar = False,grayscale = True)

        image = Image(self.projections,0,segment_settings)

        zstack = Zstack([image],self.file_id,self.series_index,self.time_index)
        zstack.physical_size = self.physical_size
        return zstack 

    def to_pdf(self):
        '''
        Convert z_stack into a pdf that displays information about how each channel was segmented
        '''
        if VERBOSE: print("Making PDF from z_stack "+self.id_z_stack) 
        processed_folder = os.path.join(TEMP_FOLDER,"z_stack_processed",self.id_z_stack)
        os.makedirs(processed_folder,exist_ok=True)
        

        pdf_imgs = []
        pdf_imgs_channel = []
        for i in range(len(self.images[0].channels)+1): 
            pdf_imgs.append([])
            pdf_imgs_channel.append(None)
        x = 0
        y = 0
        x_vars = ["channel_id","type","auto_max"]
        y_vars = ["z_index"]
        image_vars = ["file_id"]
        data = None
        for i in self.images: 
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            z_index = i.z_index
            for j in i.channels:
                channel_id = j.id_channel 
                file_id = j.id_channel
                data = {"z_index":i.z_index,"file_id":j.file_id,"channel_id":j.id_channel,"type":None,"auto_max":False}
                
                img_for_viewing_path = os.path.join(self.img_for_viewing_folder,j.file_id+".png")
                img_view = j.get_img_for_viewing(scale_bar=True,auto_max=False,colorize=True)
                if not cv2.imwrite(img_for_viewing_path,img_view):
                    raise ValueError("Could not save image to: "+img_for_viewing_path)

                x = 0
                data["type"] = "original"
                data["auto_max"] = False 
                pdf_imgs[j.channel_index+1].append(make_pdf.Image_in_pdf(x,y,img_for_viewing_path,data.copy(),x_vars,y_vars,image_vars,processed_folder,self.img_dim))
                
                x = 1
                data["type"] = "w_contours"
                data["auto_max"] = True
                j.make_img_with_contours(self.img_w_contour_folder,scale_bar = True,auto_max = True,colorize=True,add_contour_numbs=True)
                pdf_imgs[j.channel_index+1].append(make_pdf.Image_in_pdf(x,y,j.img_w_contour_path,data.copy(),x_vars,y_vars,image_vars,processed_folder,self.img_dim))
                
                pdf_imgs_channel[j.channel_index+1] = j.id_channel

            x = 0
            data["file_id"] = i.combined_mask.file_id
            data["type"] = "combined_mask"
            data["auto_max"] = None
            data["channel_id"] = -1
            pdf_imgs[0].append(make_pdf.Image_in_pdf(x,y,i.combined_mask.mask_path,data.copy(),x_vars,y_vars,image_vars,processed_folder,self.img_dim))
            
            x = 1
            data["type"] = "w_contours"
            data["auto_max"] = None
            data["channel_id"] = -1
            i.combined_mask.make_img_with_contours(self.img_w_contour_folder,scale_bar = False,auto_max=False,colorize=True,add_contour_numbs=True)
            pdf_imgs[0].append(make_pdf.Image_in_pdf(x,y,i.combined_mask.img_w_contour_path,data.copy(),x_vars,y_vars,image_vars,processed_folder,self.img_dim))
            
            pdf_imgs_channel[0] = i.combined_mask.id_channel

            y = y + 1
        if VERBOSE: print("")
        
        for i in range(len(pdf_imgs)): 
            save_path = os.path.join(GRAPHICAL_SEGMENTATION_FOLDER,self.id_z_stack+"_"+str(pdf_imgs_channel[i])+".pdf")
            pdf = make_pdf.Pdf(save_path,pdf_imgs[i])
            pdf.make_pdf()

    def __repr__(self):
        string = "{class_str} stack_id: {class_id} with n images: {n}".format(class_str = self.__class__.__name__,class_id = self.id_z_stack,n = len(self.images))
        return string

    def print_all(self):
        print(self)
        for i in self.images:
            i.print_all()

class Image: 

    def __init__(self,channels,z_index,segment_settings):
        '''
        Class of image

        Params
        channels         : list of Channel          : List containing each individual channel for this image
        z_index          : int                      : z_index of the image
        segment_settings : list of Segment_settings : Lost of Segment_settings objects describing how to process Image 
        '''
        global VERBOSE 
        
        for i in channels: 
            assert isinstance(i,Channel),"The list of Channel objects contains an object that is not a Channel object"
        
        for i in segment_settings:
            assert isinstance(i,Segment_settings),"The list of segment_settings contains an object that is not a Segment_settings object: "+str(type(segment_settings))

        channels.sort() 
        self.channels = channels 
        self.z_index = int(z_index) 
        self.segment_settings = segment_settings

        self.combined_mask = None 
    
    def set_combined_channel(self,z_stack_name,combined_mask_folder,combined_mask,combined_mask_path): 
        '''
        Set master channel that can be used to measure objects across channels
        
        Params
        z_stack_name         : str      : Name of z_stack 
        combined_mask_folder : str      : Folder to save combined mask in 
        combined_mask        : np.array : cv2.image, grayscale, uint8. 255 = pixels to measure.
        combined_mask_path   : str      : Path to save combined mask to 
        '''
        channel_index = -1
        id_channel = "Ccomb"

        empty_img = np.zeros(combined_mask.shape,dtype="uint8")
        empty_img_path = os.path.join(combined_mask_folder,z_stack_name+"_"+str(self.z_index)+"_"+id_channel+".png")
        if not cv2.imwrite(empty_img_path,empty_img):
            raise ValueError("Could not save path to: "+empty_img_path)
        
        categories = self.channels[0].categories

        self.combined_mask = Channel(empty_img_path,channel_index,self.z_index,(255,255,255),categories)
        self.combined_mask.id_channel = id_channel
        self.combined_mask.mask = combined_mask
        self.combined_mask.mask_path = combined_mask_path
    

    def make_combined_channel(self,z_stack_name,combined_mask_folder):
        '''
        Merge masks into a channel. Useful for determining objects that other structures are inside. 
        Merges channels that are marked to merge in segment_settings. 
        
        Params
        z_stack_name         : str : Name of z_stack 
        combined_mask_folder : str : Folder to save combined mask in 
        '''
        channels_to_combine = []
        for s in self.segment_settings: 
            if s.combine: 
                channels_to_combine.append(s.channel_index)
        
        combined_mask = None
        for c in self.channels: 
            if c.channel_index in channels_to_combine:
                if combined_mask is None: 
                    combined_mask = c.get_mask().copy()
                else: 
                    combined_mask = cv2.bitwise_or(combined_mask,c.get_mask())
        
        combined_mask_path = os.path.join(combined_mask_folder,z_stack_name+"_z"+str(self.z_index)+".png")
        if not cv2.imwrite(combined_mask_path,combined_mask):
            raise ValueError("Could not save path to: "+combined_mask_path)
        
        self.set_combined_channel(z_stack_name,combined_mask_folder,combined_mask,combined_mask_path)

    def make_masks(self,mask_save_folder):
        '''
        Make masks for all channels with the correct segmentation_settings from segment_settings

        Params
        mask_save_folder : str : Path to folder where masks are saved 
        '''
        for i in self.channels:
            made_mask = False
            for s in self.segment_settings: 
                if s.channel_index == i.channel_index: 
                    i.make_mask(s,mask_save_folder)
                    made_mask = True 
                    break

            if not made_mask: 
                raise ValueError("Did not find a channel index in segment settings that matched the channel.")
        return None 

    def find_z_overlapping(self,next_z): 
        '''
        Find all contours that overlap with object in next z_plane in the same channel 

        Params
        next_z : Image : Next z plane 
        '''
        for i in range(len(self.channels)): 
            if not self.channels[i].channel_index == next_z.channels[i].channel_index: 
                raise ValueError("You can't compare two images that does not have similar channel setups")
            self.channels[i].find_z_overlapping(next_z.channels[i])
        
        if self.combined_mask is not None: 
            self.combined_mask.find_z_overlapping(next_z.combined_mask)

        return None 

    def get_img_dim(self):
        #Get image dimensions 
        img_dim = self.channels[0].img_dim 
        return img_dim 

    def get_physical_res(self,xml_folder):
        #Get physical resoluion of image in um per pixel 
        return self.channels[0].get_physical_res(xml_folder)
         
    def is_inside_combined(self):
        #For all contours in Channels, check if they are inside combined mask
        for i in self.channels: 
            i.is_inside_combined(self.combined_mask)
        return None 

    def print_all(self):
        print("\t",end="")
        print(self)
        for i in self.channels: 
            i.print_all()
        return None
    
    def __lt__(self,other):
        return self.z_index < other.z_index

    def __repr__(self):
        string = "{class_str} z_index: {class_id} with n channels: {n}".format(class_str = self.__class__.__name__,class_id = str(self.z_index),n = len(self.channels))
        return string

class Channel: 
    def __init__(self,full_path,channel_index,z_index,color,categories = None,global_max=None):
        '''
        Object that represents a single channel in an image. 

        Params
        full_path     : str           : The whole path to image 
        channel_index : int           : index of channels. Must start at 0 for first channel and iterate +1
        z_index       : int           : z_index of current channel 
        color         : (int,int,int) : 8-bit color of current channel in BGR format
        categories    : Categories    : Categories relevant for this image
        global_max    : int           : global max to scale image to before changing it to 8-bit. If set to None images are assumed to be 8-bit 
        '''
        global VERBOSE 
        
        if not os.path.isfile(full_path): 
            raise ValueError("Trying to make an Channel object with file that does not exist: "+full_path)

        self.full_path = full_path 
        self.root_path,self.file_name = os.path.split(full_path)
        self.file_id = os.path.splitext(self.file_name)[0]
        if self.file_id.endswith(".ome"):
            self.file_id = self.file_id[:-4]

        self.channel_index = int(channel_index)
        self.id_channel = "C"+str(channel_index)
        self.z_index = z_index

        self.x_res = None
        self.x_unit = None
        self.color = color
        
        self.categories = categories

        self.global_max = global_max
        
        self.read_image()
        self.img_dim = self.image.shape 
        self.img_box = Rectangle([0,0,self.img_dim[1],self.img_dim[0]])
        self.mask = None
        self.mask_path = None
        self.contours = None 
        self.n_contours = None

        self.contour_groups = None

        self.img_w_contour_path = None 
        
        self.annotation_this_channel = None
        self.annotation_other_channel = []
    
    def filter_w_mask(self,mask): 
        '''
        Filter channel so that only pixels within mask are kept

        mask : Mask : Pixels to keep
        '''
        img = self.get_image()
        self.image = cv2.bitwise_and(img,img,mask = mask.mask)
        self.file_id = self.file_id + "_"+mask.mask_name
    
    def read_image(self): 
        '''
        Read image from image path and store in self.image variable
        '''
        image = cv2.imread(self.full_path,cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)
        
        if image is None: 
            raise RuntimeError("Could not read image: "+self.full_path)
        
        if self.global_max is not None: 
            image = cv2.convertScaleAbs(image, alpha=(255.0/self.global_max))
            #if VERBOSE: print("global_max is set to "+str(self.global_max)+" using that and converting image to dtype: "+str(image.dtype))
        
        self.image = image 

    def get_image(self):
        if self.image is not None: 
            return self.image 
        else:
            self.read_image()
            return self.image 

    def add_annotation(self,annotations,match_file_id=False): 
        '''
        Add annotational information to channel

        annotations    : list of Annotation : List of annotations for this z_stack  
        match_channels : bool               : If true, looking for identical file id instead of finding similar channel. 
        '''
        if match_file_id: 
            for a in annotations:
                if a.file_id == self.file_id: 
                    if a.manually_reviewed: 
                        if not self.annotation_this_channel is None: 
                            raise ValueError("Multiple annotations match file id! "+self.file_id)
                        self.annotation_this_channel = a
        else: 
            for a in annotations: 
                if self.contour_groups is not None: 
                    a.add_contour_groups(self.contour_groups)
                
                if a.id_channel == self.id_channel:
                    if not self.annotation_this_channel is None: 
                        raise ValueError("Multiple annotations match file id! "+self.file_id)
                    if a.manually_reviewed: 
                        self.annotation_this_channel = a
                else:
                    self.annotation_other_channel.append(a)
            
        if VERBOSE: 
            if self.annotation_this_channel is None: 
                print("Did not find manually reviewed annotation file.",end="\t")
            else: 
                print("Found annotation file: "+self.annotation_this_channel.file_id+"\t n_annotations: "+str(len(self.annotation_this_channel.df.index)),end="\t")
            print("And added "+str(len(self.annotation_other_channel))+" other channel annotations")
        
    def split_on_annotations(self): 
        '''
        Split up contours based on annotations
        '''
        if self.annotation_this_channel is not None: 
            self.check_annotation(self.annotation_this_channel)
            img_contours = np.zeros(self.img_dim,dtype="uint8")
            img_centers = img_contours.copy()

            contours = []
            for c in self.contours:
                if len(c.annotation_this_channel_id)>0: 
                    contours.append(c.points)
            cv2.drawContours(img_contours,contours,-1,(255),thickness = cv2.FILLED) 
            
            df = self.annotation_this_channel.df 
            for i in df.index:
                xy = (int(df.loc[i,"center_x"]),int(df.loc[i,"center_y"]))
                cv2.circle(img_centers,xy,2,color=255,thickness=-1)
            contours = self.watershed(img_contours,img_centers)
            
            new_contours = []
            for c in contours: 
                new_contours.append(Contour(c,self.img_dim,self.z_index,self.img_box))
            
            self.contours = new_contours
            self.n_contours = len(self.contours)
    
    def find_distance_centers(self,erode = None,halo = 10,min_size = 9): 
        # Find distance center of all contours
        for c in self.contours: 
            c.find_distance_centers(erode,halo,min_size)

    def split_contours(self,erode = True,halo = 10,min_size = 9): 
        '''
        Split contours with watershedding. Use halo, erode and min_size to find true centers

        Params
        erode    : bool : Whether to erode before splitting 
        halo     : int  : How many pixels from the edge to remove to find definite centers
        min_size : int  : Objects smaller than this will be removed 
        '''
        if erode == True: 
            kernel = np.ones((3,3),np.uint8)
        else: 
            kernel = None 
        
        self.find_distance_centers(kernel,halo,min_size)
        
        only_zeros = np.zeros(self.img_dim,dtype="uint8")

        contours = []
        for c in self.contours: 
            contours.append(c.points)
        img_contours = only_zeros.copy()
        cv2.drawContours(img_contours,contours,-1,(255),thickness = cv2.FILLED)

        img_centers = only_zeros
        for c in self.contours: 
            for center in c.distance_centers: 
                cv2.rectangle(img_centers,(center[0]-2,center[1]-2),(center[0]+2,center[1]+2),color = 255)

        contours = self.watershed(img_contours,img_centers)
        
        new_contours = []
        for c in contours: 
            new_contours.append(Contour(c,self.img_dim,self.z_index,self.img_box))
        
        self.contours = new_contours

    @classmethod
    def watershed(self,contours_full,contours_centers):	
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
        single_contour = np.zeros(contours_centers.shape,dtype="uint8")
        contours = []
        for i in range(2,markers.max()+1):
            single_contour[markers==i] = 255
            contour_i, hierarchy = cv2.findContours(single_contour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            single_contour[markers==i] = 0
            contours = contours + contour_i
        
        return contours 

    def check_annotation(self,other_channel = False): 
        '''
        Check whether contours contain annotational information

        Params 
        other_channel : bool : add annotation from other channels as well  
        '''
        for c in self.contours: 
            c.check_annotation(self.annotation_this_channel)
            if other_channel: 
                for a in self.annotation_other_channel: 
                    c.check_annotation(a,True)

    def get_part_of_image(self,rectangle): 
        '''
        Return only part of image instead of the whole thing. NB! the returned value is not necessarily a copy 

        Params
        rectangle : Rectangle : Part of image to return 

        Returns
        img_cropped : np.array : cropped cv2 image 
        '''
        img = self.get_image()
        return Channel.imcrop(img,rectangle)
    
    @classmethod 
    def imcrop(self,img,rectangle,value=255):
        """
        Crop image with black border if it is outside image
        
        Params
        img       : numpy array   : image
        rectangle : Rectangle     : Rectangle 
        """
        x1 = rectangle.top_left.x
        y1 = rectangle.top_left.y
        x2 = rectangle.bottom_right.x
        y2 = rectangle.bottom_right.y 
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = Channel.pad_img_to_fit_bbox(img, x1, x2, y1, y2,value=value)
        return img[y1:y2, x1:x2]

    @classmethod 
    def pad_img_to_fit_bbox(self,img, x1, x2, y1, y2,value=255):
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

    def get_img_for_viewing(self,scale_bar,auto_max,colorize,grayscale = False): 
        '''
        Modify image for viewing purpouses. 

        Params
        scale_bar : bool : Add scale bar to image 
        auto_max  : bool : Make the highest intensity pixel the highest in image
        colorize  : bool : Convert image to color as specified in channel color
        grayscale : bool : if True, do not convert image to three channel 
        
        Returns
        img                    : np.array : cv2 image after processing.
        '''
        img = self.get_image()
        
        if auto_max:
            max_pixel = np.max(np.amax(img,axis=0))+1
            img = (img/(max_pixel/255)).astype('uint8')
        
        if colorize:
            img = self.gray_to_color(img,self.color)
        else: 
            if not grayscale: 
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        if scale_bar: 
            if self.x_res is None or self.x_unit is None: 
                raise RuntimeError("Channel.x_res and Channel.x_unit must be set outside class initiation to increase performance. Try Channel.get_physical_res()")
            img = self.add_scale_bar_to_img(img,self.x_res,self.x_unit)
        
        return img
   
    def make_img_with_contours(self,img_w_contour_folder,scale_bar=False,auto_max=True,colorize=False,add_distance_centers=False,add_contour_numbs=False,scale = None):
        '''
        Make a version of the image with contours and the contour id 

        Params
        img_w_contour_folder : str   : Path to folder to save resulting image in 
        scale_bar            : bool  : add scale bar to image
        auto_max             : bool  : auto_max image
        colorize             : bool  : Convert from grayscale to channel specific color
        add_annotation       : bool  : Add annotational information
        add_contour_numbs    : bool  : Add id_contour 
        scale                : float : Scale size of image with this amount
        '''
         
        img = self.get_img_for_viewing(scale_bar,auto_max,colorize).copy()
        #img = cv2.cvtColor(self.get_image(),cv2.COLOR_GRAY2BGR)
        
        if scale is not None: 
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img,(new_width,new_height),interpolation = cv2.INTER_AREA)
        else: 
            scale = 1

        names,colors = self.categories.get_info_text_img()
        for i in range(len(names)):
            cv2.putText(img,names[i],(0,int(30*i*scale+50)),cv2.FONT_HERSHEY_SIMPLEX,1,colors[i],1,cv2.LINE_AA) 
        
        for c in self.contours:
            name = c.annotation_this_channel_type
            if len(set(name)) == 1: 
                name = name[0]
                color = self.categories.get_color(name,return_type = "bgr")
            else: 
                name = None 
                color = (255,255,255)
            
            scaled_points = Contour.scale_contour(c.points,scale)
            cv2.drawContours(img,[scaled_points],contourIdx = -1,color = color,thickness = round(1*scale))
           
            if add_distance_centers: 
                color_center = color
                for i in range(len(c.distance_centers)):
                    if i>0: 
                        color_center = (0,0,0)
                    center = (int(c.distance_centers[i][0]*scale),int(c.distance_centers[i][1]*scale))
                    cv2.circle(img,center,3,color=(255,255,255),thickness=-1)
                    cv2.circle(img,center,2,color=color_center,thickness=-1)

            if add_contour_numbs:
                all_ids = ",".join(c.annotation_this_channel_id)
                xy = (int(c.data["centroid_x"]*scale),int(c.data["centroid_y"]*scale))
                cv2.putText(img,all_ids,xy,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

        img_w_contour_path = os.path.join(img_w_contour_folder,self.file_id+".png")
        if not cv2.imwrite(img_w_contour_path,img):
            raise ValueError("Could not save image to: "+img_w_contour_path)
        
        self.img_w_contour_path = img_w_contour_path 

    @classmethod
    def gray_to_color(self,gray_img,bgr): 
        '''
        Convert grayscale image to rgb image in the color specified
        Params
        gray_img : numpy.array  : grayscale cv2 image with values 0 -> 255
        bgr      : int tupple : bgr color (b,g,r) where 0 <= b,g,r <= 255
        Returns
        color    : np.array    : BGR image
        '''
        color = np.zeros((gray_img.shape[0],gray_img.shape[1],3),np.uint8)
        color[:] = bgr
        gray_img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR)
        gray_img = gray_img.astype('float') / 255
        out = np.uint8(gray_img*color)
        
        return out

    @classmethod 
    def add_scale_bar_to_img(self,image,x_res,unit):
        '''
        Add scale bar to image
        
        Params
        image : np.array : cv2 image to add scale bar to
        x_res : float    : Number of units each pixel in x-direction corresponds to
        unit  : str      : unit of x_res
        '''
        image = image.copy()
        shape = image.shape 
        img_width = shape[1]
        img_height = shape[0]

        scale_in_pixels = int(img_width * 0.2)
        scale_in_unit = scale_in_pixels * x_res
        
        scale_in_unit_str = ("{:.0f} {unit}").format(scale_in_unit,unit=unit)
        
        scale_height = int(scale_in_pixels * 0.05)
        margin = img_width*0.05
        x1 = int(img_width - (scale_in_pixels+margin))
        y1 = int(img_height*0.9)
        scale_box = [x1,y1,x1+scale_in_pixels,y1+scale_height] #[x1,y1,x2,y2] with x0,y0 in left bottom corner

        cv2.rectangle(image,(scale_box[0],scale_box[1]),(scale_box[2],scale_box[3]),(255,255,255),thickness = -1)

        text_x = scale_box[0] + int(img_width*0.02)
        text_y = scale_box[1] - int(img_height*0.02)
        
        font_size = img_width/1000  
        
        cv2.putText(image,scale_in_unit_str,(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,font_size,(255,255,255),2)

        return image
   
    def get_mask_as_file(self,mask_save_folder): 
        if self.mask_path is not None: 
            return self.mask_path
        else: 
            self.mask_path = os.path.join(mask_save_folder,str(self.file_id)+".png")
            saved_image = cv2.imwrite(self.mask_path,self.mask)
            if saved_image is False: 
                print("Could not save image to: "+self.mask_path)
            return self.mask_path

    def get_mask(self):
        if self.mask is not None: 
            return self.mask
        elif self.mask_path is not None: 
            mask = cv2.imread(self.mask_path,cv2.IMREAD_GRAYSCALE)
            return mask
        else: 
            raise RuntimeError("Need to create mask before it can be returned")
    
    def get_part_of_mask(self,rectangle): 
        '''
        Return only part of mask instead of the whole thing. NB! The returned matrix is not a copy! 

        Params
        rectangle : Rectangle : Part of mask to return 

        Returns
        img_cropped : np.array : cropped cv2 image 
        '''
        img = self.get_mask()
        return Channel.imcrop(img,rectangle)
    
    @classmethod 
    def remove_small_objects(self,image,min_size):
        '''
        Remove all objects that is smaller than min_size
        
        Params
        image    : np.array : cv2 image that is black and white
        min_size : int      : minimum size of object to remove

        Returns 
        img2     : np.array : cv2 image with small objects removed
        '''
        if min_size == 0: 
            return image
        
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        img2 = np.zeros((output.shape),np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        return img2

    def make_mask(self,segment_settings,mask_save_folder = None):
        '''
        Generate a mask image of the channel where 255 == true pixel, 0 == false pixel 

        Params
        segment_settings : Segment_settings : Object containing all the segment settings
        mask_save_folder : str              : where to save the mask. If none, masks are not saved. 
        
        '''
        
        img = self.get_image().copy()
        assert img is not None, "Can't make mask out of Image == None"
        
        s = segment_settings 
        
        if s.contrast is not None: 
            if not (0.99 < s.contrast < 1.001 ):
                img = (img*s.contrast).astype('uint8')
        
        if s.auto_max is not None: 
            if s.auto_max: 
                max_pixel = np.max(np.amax(img,axis=0))
                img = (img/(max_pixel/255)).astype('uint8')
        
        if s.thresh_type is not None and not (str(s.thresh_type)+" " == "nan "): 
            if s.thresh_type == "canny_edge": 
                img = cv2.Canny(img,s.thresh_lower,s.thresh_upper)
            elif s.thresh_type == "binary":
                ret,img = cv2.threshold(img,s.thresh_lower,s.thresh_upper,cv2.THRESH_BINARY)
            else:
                raise ValueError(str(thresh_type)+" is not an available threshold method")
            
            if type(s.open_kernel) is np.ndarray:  
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, s.open_kernel)
                
            if type(s.close_kernel) is np.ndarray:  
                img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,s.close_kernel)
                
            if s.min_size is not None: 
                img = Channel.remove_small_objects(img,s.min_size)
        
        assert img is not None, "Mask is none after running function. That should not happen"
        
        self.mask = img
        if mask_save_folder is not None:
            self.get_mask_as_file(mask_save_folder)
    
    def find_contours(self,min_contour_area= 5): 
        '''
        Find contours in masked image of channel 

        Params
        min_contour_area : int : Contours smaller than this are filtered out 
        '''
        mask = self.get_mask()
        contours_raw, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        contours = []
        for i in range(len(contours_raw)):
            M = cv2.moments(contours_raw[i])
            contour_area = M['m00']
            if contour_area > min_contour_area:
                contours.append(Contour(contours_raw[i],self.img_dim,self.z_index,self.img_box))
        
        self.contours = contours
        self.n_contours = len(self.contours) 
    
    def update_contour_stats(self): 
        #update contour stats for all contours in image
        for c in self.contours: 
            c.update_contour_stats()

    def group_contours(self,n_div=200):
        '''
        Group contours into areas they overlap so you don't need to check all contours with all contours 

        Params
        n_div : int : number of rectangles to convert image into
        '''
        image_dim = self.get_image().shape 
        image_dim = (image_dim[0],image_dim[1])
        self.contour_groups = Contour_groups(image_dim,n_div)
        
        for c in self.contours:
            self.contour_groups.add_contour(c)
    
    def get_group_contours(self,group_names): 
        if self.contour_groups is None: 
            if VERBOSE: print("Contours not grouped, using whole list")
            return self.contours
        else: 
            return self.contour_groups.get_contours_in_group(group_names)
    
    def find_z_overlapping(self,next_z):
        '''
        Find all contours in next z plane in same channel that overlaps with this object

        Params
        next_z : Channel : image channel with contours to compare against
        '''
        for i in self.contours: 
            i.find_z_overlapping(next_z)
            
    def measure_channels(self,channels): 
        '''
        Measure intensity of supplied channels in all contours in this channel
        
        Params
        channel : list of Channel : Channels to measure intensities in 
        '''
        
        for i in self.contours: 
            i.measure_channel(channels)
    
    def write_single_objects(self,single_objects_folder,box_size = 120,merge_categories=None,keep_ambigous=False,dry_run = False):
        '''
        Write images of only objects on a white background with a folder for each category
        
        Params
        single_objects_folder : str  : path to folder to write objects in
        box_size              : int  : Size of box around object
        merge_categories      : dict : If not None, merge category from key into value in dictionary
        keep_ambigous         : bool : If object has more than one category it is ambigous, default is to throw them out. 
        dry_run               : bool : Do everything except saving images. 
        '''
        # Get objects, their names and their categories
        objects = []
        objects_categories = []
        n_unassigned = 0
        n_ambigous = 0
        n_assigned = 0
        for c in self.contours:
            only_object = c.get_only_object(self,box_size = 120)
            objects.append(only_object)
            category = c.annotation_this_channel_type
            if len(category) == 0:
                category = "None"
                n_unassigned += 1
            elif len(category) == 1:
                category = category[0]
                n_assigned += 1
            elif len(category)>=2:
                category = "Ambigous"
                n_ambigous += 1
            objects_categories.append(category)
        
        if VERBOSE: print("Total objects: \t"+str(len(objects)))
        if VERBOSE: print("n_unassigned: \t"+str(n_unassigned)+"\nn_assigned: \t"+str(n_assigned)+"\nn_ambigous: \t"+str(n_ambigous))
        
        objects_fnames = []
        for i in range(len(objects)):
            objects_fnames.append(self.file_id+"_"+str(i)+".png")
        
        # Make df representing objects and where they should be put
        df = pd.DataFrame(data={'category':objects_categories,'fname':objects_fnames,'object':objects})
        if not keep_ambigous:
            old_length = len(df.index)
            df = df.loc[df["category"] != "Ambigous",]
            if VERBOSE: print("Threw out objects labelled as 'Ambigous'.\tOld length: "+str(old_length)+"\tnew length: "+str(len(df.index)))
        
        if merge_categories is not None: 
            for key in merge_categories.keys():
                if VERBOSE: old_length = sum(df["category"]==key)
                df.loc[df["category"]==merge_categories[key],"category"] = key
                if VERBOSE: 
                    new_length = sum(df["category"]==key)
                    print("Merged '"+merge_categories[key]+"' into '"+key+"'.\tOld number of objects in category "+key+": "+str(old_length)+"\tnew number: "+str(new_length))
        
        categories = df['category'].unique()
        if VERBOSE: print("Categories: "+str(categories))
        
        if VERBOSE: print(df['category'].value_counts())
        for i in df.index: 
            df.loc[i,"out_path"] = os.path.join(single_objects_folder,"train",df.loc[i,"category"],df.loc[i,"fname"])
    
        # Write out folders and files
        for c in categories:
            os.makedirs(os.path.join(single_objects_folder,"train",c),exist_ok=True)
        
        #if VERBOSE: print(df)
        
        if dry_run:
            if VERBOSE: print("Dry run. Not writing images.")
        else: 
            counter = 0
            tot_length = len(df.index)
            for i in df.index: 
                if VERBOSE: 
                    counter += 1
                    if counter % 500 == 0: 
                        print(str(counter)+"/"+str(tot_length)+"\t Writing image: "+df.loc[i,"out_path"])
                if not cv2.imwrite(df.loc[i,"out_path"],df.loc[i,"object"]):
                    raise ValueError("Could not write image to "+df.loc[i,"out_path"])
        
    def classify_objects(self,ai_predict): 
        '''
        Classify contour objects

        Params
        ai_predict : AI_predict : instance of object with functions to predict object class
        '''
        
        objects = []
        for c in self.contours: 
            only_object = c.get_only_object(self,box_size = 120)
            objects.append(only_object)

        if len(objects)>0: 
            predictions = ai_predict.get_predictions(objects)
        
            for i in range(len(predictions)):
                self.contours[i].annotation_this_channel_type = [predictions[i]]
                self.contours[i].annotation_this_channel_id = [str(i)]

    def get_contour_stats(self):
        '''
        Get a pandas data frame containing the stats of all the contours in the image

        Returns
        df : pd.DataFrame : information about contours 
        '''
        df = pd.DataFrame()
        for c in self.contours: 
            temp = pd.Series(c.data)
            temp["contour_id"] = c.id_contour 
            df = df.append(temp,ignore_index = True)
        
        df["z_index"] = self.z_index 
        df["channel_id"] = self.id_channel 

        return df 

    def write_annotation_file(self,annotation_folder,add_to_changelog = None): 
        '''
        Write annotation file 

        Params
        annotation_folder : path : folder to put file in 
        add_to_changelog  : str  : string to add to changelog 
        '''

        now = datetime.datetime.now()
        today = now.strftime("%Y-%m-%d-%H:%M")

        out_path = os.path.join(annotation_folder,self.file_id + ".txt")
        
        df = self.get_contour_stats()
        if self.annotation_this_channel is not None: 
            reviewed_by_human = self.annotation_this_channel.manually_reviewed
            changelog = self.annotation_this_channel.changelog
            next_object_id = self.annotation_this_channel.next_object_id
        else: 
            reviewed_by_human = False 
            changelog = ""
            next_object_id = None
        
        if add_to_changelog is not None: 
            changelog = changelog + today +"\t"+add_to_changelog + "\n"
        
        info.Annotation.write_annotation_file(out_path,reviewed_by_human,changelog,df,next_object_id)

    def print_all(self):
        print("\t\t",end="")
        print(self)
        #if self.contours is not None: 
        #    for i in self.contours: 
        #        i.print_all()
    
    def get_physical_res(self,xml_folder):
        '''
        Find physical resolution from ome.tiff file

        Returns
        physical_size_x : dict : dictionary of resolutions of "x","y","z" and the "unit".  

        '''
        def physical_size_as_1(physical_size): 
            physical_size["x"] = 1
            physical_size["y"] = 1
            physical_size["z"] = 1
            physical_size["unit"] = "px" 
            return physical_size

        global WARNINGS

        physical_size = {"x":None,"y":None,"z":None,"unit":None}
        
        if "ome.tiff" not in self.file_name:
            if WARNINGS:
                print("Warning: provided file is not in .ome.tiff format and the pysical resolutions is set to 1 pixel per pixel")
            return physical_size_as_1(physical_size)

        xml_path = os.path.join(xml_folder,self.file_id+".xml")

        cmd = "tiffcomment {image} > {xml}".format(image = self.full_path,xml = xml_path)
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise ValueError("Command did not exit properly: "+cmd)
        
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
                        physical_size["x"] = float(value)

                    if name == "PhysicalSizeY":
                        physical_size["y"] = float(value)

                    if name == "PhysicalSizeZ":
                        physical_size["z"] = float(value)
                    
                    if name == "PhysicalSizeXUnit":
                        x_unit = value
                    
                    if name == "PhysicalSizeYUnit":
                        y_unit = value
                    
                    if name == "PhysicalSizeZUnit":
                        z_unit = value
        
        if physical_size["x"] is None or physical_size["y"] is None or physical_size["z"] is None: 
            if WARNINGS: 
                print("Warning: File did not contain information about resolution for all axis and the pyiscal resolution is set to 1 pixel per pixel. Info i found in file: "+str(physical_size))
            return physical_size_as_1(physical_size) 
        
        all_units = [x_unit,y_unit,z_unit]
        for i in all_units: 
            if i is None: 
                if WARNINGS: 
                    print("Warning: one of the units where not found and the physical resolution is set to 1 pixel per pixel. units: "+str(all_units))
                return physical_size_as_1(physical_size)
        
        if len(set(all_units)) != 1:
            if WARNINGS: 
                print("Warning: All units where not the same! They need to be. Setting physical resolution to 1 pixel per pixel. All units: "+str(all_units))
            return physical_size_as_1(physical_size)
        
        #Try to catch the most common weird characters
        unit = x_unit 
        if VERBOSE: print("Unit raw: "+str(unit)+", as unicode: "+str(unit.encode()),end = " ")
        unit = unit.replace("\u03Bc","u") #unicode for greek mu symbol
        unit = unit.replace("µ","u") #micro symbol
        if VERBOSE: print(" After fixing: "+str(unit))

        if WARNINGS: 
            is_ascii_str = lambda s: len(s) == len(s.encode())
            if not is_ascii_str(unit): 
                print("Warning: One of the characters in unit is not compatible with ASCII and you might get some problems downstream. unit = "+str(unit))

        physical_size["unit"] = unit

        if VERBOSE:
            print("For object: \""+str(self)+"\" I found physical size: "+str(physical_size))

        return physical_size
    
    def is_inside_combined(self,other_channel):
        '''
        For all contours in channel, find those contours which it is in other_channel 
        
        Params
        other_channel : Channel : Channel object which contours to check if contours in this channel is inside 
        '''
        assert self.contours is not None,"Must run Channel.find_contours() before contours can be evaluated"
        assert other_channel.contours is not None,"Must to run Channel.find_contours() before contours can be evaluated"
        for i in self.contours:
            i.find_is_inside(other_channel)

    def __lt__(self,other):
        return self.channel_index < other.channel_index 
    
    def __repr__(self):
        string = "{class_str} channel_index: {class_id} with n contours: {n}".format(class_str = self.__class__.__name__,class_id = self.id_channel,n = self.n_contours)
        return string

class Contour:

    def __init__(self,points,img_dim,z_index,img_box):
        '''
        Object with information about contour found in channel 

        Params
        points  : cv2.contour   : points[point][0][x][y] x and y relative to top_left corner 
        img_dim : tupple of int : (y,x) x and y relative to top_left corner
        z_index : int           : z_index of contour
        img_box : Rectangle     : Rectangle of image to check if at edge
        '''
        global VERBOSE 
        self.points = points 
        self.img_dim = img_dim 
        self.img_box = img_box

        global contour_id_counter
        self.id_contour = contour_id_counter 
        contour_id_counter = contour_id_counter + 1
        
        self.z_index = z_index
        self.next_z_overlapping = None 
        self.prev_z_overlapping = [] 
        self.is_inside = None
        
        self.contour_box = self.get_contour_rectangle()
        self.contour_mask = None

        self.group_name = []

        self.data = None

        self.manually_reviewed = False
        self.annotation_this_channel_type = []
        self.annotation_this_channel_id   = [] 
        self.annotation_other_channel_type = []
        self.annotation_other_channel_id   = [] 
        
        self.distance_centers = None
    
    def get_contour_rectangle(self):
        '''
        Calculate the bounding box around the contour and return the coordinates as a rectangle object
        
        Returns
        rectangle : Rectangle : Rectangle object that contains the contour  
        '''
        x,y,w,h = cv2.boundingRect(self.points)
        box = [x,y,x+w,y+h]
        return Rectangle(box)
    
    def get_contour_mask_whole_img(self):
        '''
        Draws the contour as a white mask on a black background the size of the entire image 

        Returns 
        contour_mask : np.array : cv2 image with only this contour
        '''
        contour_mask = np.zeros(self.img_dim,dtype="uint8")
        cv2.drawContours(contour_mask,[self.points],-1,color=255,thickness = -1)
        return contour_mask
    
    def get_contour_mask(self): 
        if self.contour_mask is None: 
            self.contour_mask = self.make_contour_mask()
        return self.contour_mask

    def make_contour_mask(self):
        '''
        Draws the contour as a white mask on a black background just the size bounding box around the contour. 

        Returns 
        contour_mask : np.array : cv2 image with only this contour
        '''
        new_points = self.points_new_origo(self.contour_box.top_left)
        img_dim = self.contour_box.get_height_and_width()
        
        contour_mask = np.zeros(img_dim,dtype="uint8")
        cv2.drawContours(contour_mask,[new_points],-1,color=255,thickness = -1)
        return contour_mask   

    def get_contour_mask_other_box(self,other_box):
        '''
        Draw the contour as a white mask on a black background. The size of the image will equal the provided other_box and parts of the contour outside this one will be cropped away. 

        Params
        other_box    : Rectangle : Box to draw the other contour inside 

        Returns
        contour_mask : np.array : cv2 image with only this contour 
        '''
        bigger_rectangle = self.contour_box.bigger_rectangle(other_box) 

        img_big_box = np.zeros(bigger_rectangle.get_height_and_width(),dtype="uint8")
        new_points = self.points_new_origo(bigger_rectangle.top_left)
        cv2.drawContours(img_big_box,[new_points],-1,color=255,thickness = -1)
        
        crop_box = other_box.new_origo(bigger_rectangle.top_left)
        img_other_box = img_big_box[crop_box.top_left.y:crop_box.bottom_right.y,crop_box.top_left.x:crop_box.bottom_right.x]

        return img_other_box
    
    def points_new_origo(self,new_origo,subtract = True): 
        #wrapper for Contour.new_origo()
        return Contour.new_origo(self.points,new_origo,subtract = subtract)

    @classmethod 
    def scale_contour(self,contour,scale):
        '''
        Get new points list of cv2 contour that are scaled to a new size  

        Params
        contour : cv2.Contour : Contour that should be scaled 
        scale   : float       : Factor to scale with 
        '''
        if scale == 1: 
            return contour
        new_points = contour.copy()
        for i in range(len(new_points)): 
            new_points[i][0][0] = new_points[i][0][0] * scale   
            new_points[i][0][1] = new_points[i][0][1] * scale  
        
        return new_points 

    @classmethod 
    def new_origo(self,contour,new_origo,subtract = True):
        '''
        Get new points list of cv2 contour with a new origo point  

        Params
        contour   : cv2.Contour : Contour that should get a new origo 
        new_origo : Point       : coordinates of new origo 
        subtract  : bool        : Subtract origo point. Set to add instead. Enables easy reversion. 
        '''
        new_points = contour.copy()
        if subtract: 
            for i in range(len(new_points)): 
                new_points[i][0][0] = new_points[i][0][0] - new_origo.x   
                new_points[i][0][1] = new_points[i][0][1] - new_origo.y  
        else: 
            for i in range(len(new_points)): 
                new_points[i][0][0] = new_points[i][0][0] + new_origo.x   
                new_points[i][0][1] = new_points[i][0][1] + new_origo.y  
        
        return new_points 

    def find_distance_centers(self,erode=None,halo = 10,min_size = 9): 
        '''
        Find the distance centers of the contour. Could be multiple if using erode and/or halo. 

        Params
        erode    : np.ones : kernel to use for eroding
        halo     : int     : Pixel that are closer to background than this is removed before finding distance centers 
        min_size : int     : Objects smaller than this is filtered out 
        '''
        mask = self.make_contour_mask()
        
        if erode is not None: 
            mask = cv2.erode(mask,erode,iterations = 1)
        
        mask_dist = cv2.distanceTransform(mask,cv2.DIST_L2,5)
        
        if halo is not None: 
            if halo>0: 
                ret,mask = cv2.threshold(mask_dist,halo,255,cv2.THRESH_TOZERO)
                mask = np.uint8(mask)
        
        if min_size is not None: 
            mask = Channel.remove_small_objects(mask,min_size) 

        max_value = mask.argmax()
        if max_value == 0: 
            #object completely dissappeared, assume 1 contour and just use the maximum distance
            n_contours = 1
        else: 
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            n_contours = len(contours)

        if n_contours == 1: 
            center = mask_dist.argmax()
            center = np.unravel_index(center,mask_dist.shape)
            xc = center[1] + self.contour_box.top_left.x
            yc = center[0] + self.contour_box.top_left.y 
            self.distance_centers = [(xc,yc)]
        
        elif n_contours > 1: 
            out_contours = []
            only_zeros = np.zeros(mask.shape,dtype="uint8")
            self.distance_centers = []
            for c in contours:
                only_contour = cv2.drawContours(only_zeros.copy(),[c],-1,color=255,thickness = -1)
                only_contour_dist = cv2.distanceTransform(only_contour,cv2.DIST_L2,5)

                center = only_contour_dist.argmax()
                center = np.unravel_index(center,only_contour_dist.shape)
                xc = center[1] + self.contour_box.top_left.x 
                yc = center[0] + self.contour_box.top_left.y 
                
                self.distance_centers.append((xc,yc))
        else: 
            raise ValueError("Contour should not completely dissappear, but it did.")

    def is_at_edge(self):
        '''
        Check if this contour is overlapping with the edge of the image
        '''
        return self.img_box.at_edge(self.contour_box,edge_size = 1) 
    
    def measure_channel(self,channels):
        '''
        Measure this contours sum grey and sum positive pixels for all channels

        Params
        channels : list of Channel : channels to measure
        '''
        only_contour = self.get_contour_mask()
        for channel in channels:
            part_of_img = channel.get_part_of_image(self.contour_box)
            mean_grey = cv2.bitwise_and(part_of_img,part_of_img,mask=only_contour)
            mean_grey = np.sum(mean_grey)
            
            part_of_mask = channel.get_part_of_mask(self.contour_box)
            sum_pos_pixels = cv2.bitwise_and(part_of_mask,part_of_mask,mask=only_contour)
            sum_pos_pixels = np.sum(sum_pos_pixels)/255

            data = {
                "sum_grey_"+str(channel.id_channel):mean_grey,
                "sum_positive_"+str(channel.id_channel):sum_pos_pixels}
            
            self.data.update(data)
    
    def get_only_object(self,channel,box_size):
        '''
        Get only the object on a white background

        Params
        channel  : Channel : Channel to get object from
        box_size : int     : size of box to extract
        '''
       
        center = self.distance_centers[0]
        box_size = int(box_size/2)
        window = Rectangle([center[0]-box_size,center[1]-box_size,center[0]+box_size,center[1]+box_size])
        
        only_contour_mask = self.get_contour_mask_other_box(window)
        part_of_img = channel.get_part_of_image(window)
        
        part_of_img = cv2.bitwise_not(part_of_img)
        only_object = cv2.bitwise_and(part_of_img,part_of_img,mask = only_contour_mask)
        only_object = cv2.bitwise_not(only_object)
        
        return only_object

    def update_contour_stats(self):
        '''
        Update statistics about Contour 
        '''
        M = cv2.moments(self.points)
        area = M['m00']
        perimeter = cv2.arcLength(self.points,True)
        (x_circle,y_circle),radius = cv2.minEnclosingCircle(self.points)
        
        if len(self.annotation_this_channel_type) == 1:
            object_type = self.annotation_this_channel_type[0]
            object_id = self.annotation_this_channel_id[0]
        else: 
            object_type = None 
            object_id = None 
        
        if self.distance_centers is None: 
            self.find_distance_centers(None,None,None)
        
        distance_center = [None,None] 
        if self.distance_centers is not None: 
            if len(self.distance_centers) == 1: 
                distance_centers = self.distance_centers[0]

        data = {
            "img_dim_yx":str(self.img_dim),
            "area":area,
            "centroid_x":int(M['m10']/M['m00']),
            "centroid_y":int(M['m01']/M['m00']),
            "center_x": distance_center[0],
            "center_y": distance_center[1],
            "z_index":self.z_index,
            "perimeter":perimeter,
            "hull_tot_area":cv2.contourArea(cv2.convexHull(self.points)),
            "radius_enclosing_circle":radius,
            "equivalent_radius":np.sqrt(area/np.pi),
            "circularity":float(4)*np.pi*area/(perimeter*perimeter),
            "at_edge":self.is_at_edge(),
            #"next_z_overlapping":self.contour_list_as_str(self.next_z_overlapping), 
            #"prev_z_overlapping":self.contour_list_as_str(self.prev_z_overlapping), 
            "is_inside":self.contour_list_as_str(self.is_inside),
            "manually_reviewed": self.manually_reviewed,
            "type": object_type,
            "object_id": object_id,
            "annotation_this_channel_type"  : ",".join(self.annotation_this_channel_type),
            "annotation_this_channel_id"    : ",".join(self.annotation_this_channel_id),
            "annotation_other_channel_type" : ",".join(self.annotation_other_channel_type),
            "annotation_other_channel_id"   : ",".join(self.annotation_other_channel_id)
        }
        
        self.data = data 
        return None        
            
    def is_overlapping(self,other_channel,overlap_thresh,assume_overlap_thresh=36,only_max = False):
        '''
        Find all contours in other channel that are overlapping with this contour 
        
        Params
        other_channel  : Channel : Channel to check if it has any contours that this one is inside
        overlap_thresh : int     : The lowest number of overlapping pixels that are required for adding as overlapping 
        assume_overlap_thresh    : int : For a small object the difference in overlap between its bounding box and actual are is so small that we can assume overlap and not check it
        only_max       : bool    : Only return the contour that is maximally overlapping 

        Returns 
        overlapping    : list of Contour : Contours in other channel that is overlapping 
        '''
        overlapping = []
        overlapping_amount = []
        other_contours = other_channel.get_group_contours(self.group_name)
        for other_contour in other_contours: 
            overlaps = self.contour_box.overlaps(other_contour.contour_box)
            if overlaps: 
                if self.contour_box.get_area() < assume_overlap_thresh:
                    overlapping.append(other_contour) 
                else: 
                    other_contour_mask = np.zeros(self.contour_box.get_height_and_width(),dtype="uint8")
                    new_points = other_contour.points_new_origo(self.contour_box.top_left)
                    cv2.drawContours(other_contour_mask,[new_points],-1,255,-1)
                
                    this_contour_mask = self.get_contour_mask()
                    overlap_img = cv2.bitwise_and(other_contour_mask,this_contour_mask)
                    overlap_amount_i = overlap_img.sum() 
                    if overlap_amount_i>overlap_thresh:
                        overlapping.append(other_contour)
                        overlapping_amount.append(overlap_amount_i)
        
        if only_max: 
            if len(overlapping)>1: 
                max_i = overlapping_amount.index(max(overlapping_amount))
                overlapping = [overlapping[max_i]]
       
        return overlapping
        
    def find_is_inside(self,combined_mask):
        '''
        Check if any of the contours in this channel is inside combined mask

        combined_mask : Channel : Channel with contours to check if these ones are inside  
        '''
        if self.is_inside is not None: 
            raise ValueError("Contour.is_inside is already set. Did you run Contour.is_inside twice ? ")
        self.is_inside = self.is_overlapping(combined_mask,overlap_thresh=1,assume_overlap_thresh=0,only_max=True)
        if len(self.is_inside)>1: 
            print("Contour: "+str(self))
            print("is inside: "+str(self.is_inside))
            raise RuntimeError("Contour is inside multiple objects in combined mask. That should not be possible. len(self.is_inside) = "+str(len(self.is_inside)))

    def find_z_overlapping(self,next_z): 
        '''
        Check if this contour overlaps with any contour in the other channel

        Params
        next_z : Channel : The next z-plane of the same channels 

        '''
        if self.next_z_overlapping is not None: 
            raise ValueError("Contour.next_z_overlapping is already set. Did you run Contour.find_z_overlapping() twice ?")
        self.next_z_overlapping = self.is_overlapping(next_z,overlap_thresh=3)
        
        for i in self.next_z_overlapping:
            if not (self in i.prev_z_overlapping): 
                i.prev_z_overlapping.append(self)
    
    def check_annotation(self,annotation,other_channel=False):
        '''
        Look through annotation and add those categories that are inside object

        Params
        annotation    : Annotation : object containing annotation information 
        other_channel : bool       : if true, this annotation belongs to another channel and results are stored separately 
        '''
        if annotation is not None: 
            if not other_channel: 
                self.manually_reviewed = annotation.manually_reviewed 
            df = annotation.get_points(self.group_name)
            for i in df.index:
                xy = (df.loc[i,"center_x"],df.loc[i,"center_y"]) 
                if self.contour_box.contains_point(xy):
                    if cv2.pointPolygonTest(self.points,xy,False) > 0:
                        if not other_channel: 
                            self.annotation_this_channel_type.append(str(df.loc[i,"type"]))
                            self.annotation_this_channel_id.append(str(df.loc[i,"object_id"]))
                        else: 
                            self.annotation_other_channel_type.append(str(df.loc[i,"type"]))
                            self.annotation_other_channel_id.append(str(df.loc[i,"object_id"]))
                            
            if not other_channel: 
                self.annotation_this_channel_type,self.annotation_this_channel_id = set_zipped_list(self.annotation_this_channel_type,self.annotation_this_channel_id)
            else: 
                self.annotation_other_channel_type,self.annotation_other_channel_id = set_zipped_list(self.annotation_other_channel_type,self.annotation_other_channel_id)

    def get_all_z_overlapping(self,all_z_overlapping):
        '''
        Look through the contours recursively and return all contours that are overlapping in z

        Params
        all_z_overlapping : list of Contour : List of all contours that are overlapping in z_dimension 
        '''
        if self not in all_z_overlapping: 
            all_z_overlapping.append(self)
            
            for z in self.next_z_overlapping:
                z.get_all_z_overlapping(all_z_overlapping)
            
            for z in self.prev_z_overlapping: 
                z.get_all_z_overlapping(all_z_overlapping) 
        
        return all_z_overlapping
    

    @classmethod 
    def contour_list_as_str(self,contour_list):
        #convert list of contours to list of contour ids 
        if contour_list is None:
            return None
        else: 
            out_str = "" 
            for c in contour_list: 
                if out_str is "": 
                    out_str = str(c.id_contour) 
                else: 
                    out_str = out_str + "," + str(c.id_contour)
            return out_str 

    def print_all(self):
        print("\t\t\t",end="")
        print(self)

    def __repr__(self):
        string = "{class_str}, contour_index: {class_id}, n_points: {n}".format(class_str = self.__class__.__name__,class_id = self.id_contour,n = len(self.points))
        return string

class Contour_groups: 

    def __init__(self,img_dim,n_div):
        '''
        Location groups for contours in image. Each contour gets a Contour_group that is the closest to its location. 

        img_dim : tuple of int : Dimensions of image (y,x)
        n_div   : int          : Number of groups to divide it into
        '''
        self.img_dim = img_dim 

        ratio = self.img_dim[0]/self.img_dim[1]

        self.n_y = int(np.sqrt(n_div*ratio))#n_div = x*y, x = y/ratio, solving for y
        self.n_x = int(self.n_y/ratio)
        
        if self.n_y<1: 
            self.n_y = 1
        if self.n_x<1: 
            self.n_x = 1

        self.n_tot = self.n_y * self.n_x
        
        #print("img_dim: {dim},n_div: {n_div},n_y: {ny}, n_x: {nx}, ntot: {tot}".format(dim = self.img_dim,n_div=n_div,ny=self.n_y,nx=self.n_x,tot=self.n_tot))
        
        self.y_step = self.img_dim[0]/self.n_y 
        self.x_step = self.img_dim[1]/self.n_x 

        self.contour_groups = {}

        for y in range(self.n_y):
            for x in range(self.n_x):
                actual_y = int((y+0.5) * self.y_step)
                actual_x = int((x+0.5) * self.x_step)
                group = Contour_group((x,y), (self.x_step,self.y_step))
                self.contour_groups[group.name] = group

    def add_contour(self,contour):
        '''
        Find the Contour_group that is the closest to the contour

        contour : Contour : Contour to add to Contour_group 
        '''
        for key in self.contour_groups.keys(): 
            c = self.contour_groups[key]
            if c.rect.overlaps(contour.contour_box): 
                c.contours.append(contour)
                contour.group_name.append(c.name)
        
        assert len(contour.group_name) > 0,"Contour did not get a group name!"
    
    def get_contour_groups_with_point(self,xy): 
        '''
        Get all the names of contour_groups with this point inside 

        Params
        xy : tuple of float : point 

        Returns
        contour_names : str : The names of all contour_groups with this points inside, separated with "," 
        '''
        contour_groups_out = []
        for c in self.contour_groups.keys(): 
            if self.contour_groups[c].contains_point(xy): 
                contour_groups_out.append(c)
        contour_groups_out = ",".join(contour_groups_out)
        return contour_groups_out

    def get_contours_in_group(self,group_names):
        '''
        Return all contours that has group names 

        Params 
        group_names : str : list of group names

        Returns
        out_contours : list of Contour : contours belonging to the groups in input
        '''
        out_contours = []
        for g in group_names: 
            out_contours = out_contours + self.contour_groups[g].contours
        
        return set(out_contours)

    def print_matrix(self,terminal_width = 120):
        '''
        Print a pretty matrix with info about object

        terminal_width : int : maximal length width of output. Adding just dots past that
        '''
        prev_y = -1
        out_str = []
        this_line = ""
        max_width = 20 
        for key in self.contour_groups.keys():
            c = self.contour_groups[key]
            str_to_add = c.short_repr() 
            while len(str_to_add) < max_width: 
                str_to_add = str_to_add + " "
            
            if c.y != prev_y: 
                out_str.append(this_line)
                this_line = ""
                prev_y = c.y
            
            if len(this_line) < (terminal_width-5): 
                this_line = this_line + str_to_add

        for l in out_str: 
            print(l,end="")
            if len(l) >= (terminal_width-5): 
                print("...",end="")
            print("")

    def draw_rectangles(self,img): 
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,255),(255,0,255),(255,255,0)]
        colors_i = 0
        for key in self.contour_groups.keys():
            c = self.contour_groups[key]
            c.rect.draw(img,colors[colors_i])
            colors_i = colors_i + 1
            if colors_i >= (len(colors)-1):
                colors_i = 0

    def __repr__(self):
        return "{class_str}: img_dim={img_dim}, n_y={ny}, n_x={nx}, n_tot={ntot}".format(class_str = self.__class__.__name__,img_dim = self.img_dim,ny=self.n_y,nx = self.n_x,ntot=self.n_tot)

class Contour_group: 
    def __init__(self,xy_center,step_xy): 
        '''
        Class that represents one position in the image. All contours that are closest to this one belongs to this group
        
        Params
        xy_center : tupple of int   : x and y of abstract contour group 
        step_xy   : tupple of float : how much +1 in xy_center represents in actual pixels 
        '''
        self.step_xy = step_xy 

        #describing center
        self.x = xy_center[0]
        self.y = xy_center[1]
        self.actual_x = int((self.x+0.5)*self.step_xy[0]) 
        self.actual_y = int((self.y+0.5)*self.step_xy[1])

        self.name = "x"+str(self.x)+"y"+str(self.y)
        
        x1 = self.actual_x - (self.step_xy[0]/2) - 2 
        y1 = self.actual_y - (self.step_xy[1]/2) - 2
        x2 = self.actual_x + (self.step_xy[0]/2) + 2
        y2 = self.actual_y + (self.step_xy[1]/2) + 2
        self.rect = Rectangle([x1,y1,x2,y2])
        
        self.contours = []
    
    def contains_point(self,xy): 
        '''
        Check if point is inside contour group

        Params
        xy : tuple of float : Point to check if it is inside 
        '''
        return self.rect.contains_point(xy) 

    def short_repr(self): 
        return self.name + "("+str(self.actual_x)+","+str(self.actual_y)+")"

    def __repr__(self):
        n_contours = len(self.contours)
        return "{class_str}: name = {name}, x={x}, y={y}, n_contours={n}".format(class_str = self.__class__.__name__,name=self.name,x=self.x,y=self.y,n = n_contours)
        

class Point: 

    def __init__(self,x,y):
        #Point with origo in top-left corner
        self.x = x
        self.y = y

    def __repr__(self):
        return("Point([{x},{y})]".format(x=self.x,y=self.y))

class Rectangle:
    def __init__(self,points):
        '''
        Rectangle object with origo in top-left corner
        
        Params
        points : list of int : [x1,y1,x2,y2]
        '''
        self.top_left      = Point(points[0],points[1])
        self.bottom_right  = Point(points[2],points[3])
    
    def overlaps(self,other):
        '''
        Check if this rectangle overlaps with another 
        
        Params
        other    : Rectangle : Other rectangle to check
        
        Returns 
        overlaps : bool      : True if rectangles overlap
        '''
        return not (self.top_left.y > other.bottom_right.y or self.top_left.x > other.bottom_right.x or self.bottom_right.y < other.top_left.y or self.bottom_right.x < other.top_left.x)
    
    def is_inside(self,other): 
        '''
        check if this rectangle is entierly encompassed in the other rectangle

        Params
        other : Rectangle : Other rectangle to check if is inside 
        '''
        for p in [other.top_left,other.bottom_right]: 
            if not self.contains_point(p): 
                return False
        return True

    def contains_point(self,xy): 
        '''
        Check if point is inside rectangle 

        Params
        xy : tuple of float : Point to check if is inside (x,y)
        '''
        return (xy[0] > self.top_left.x and xy[0] < self.bottom_right.x and xy[1] > self.top_left.y and xy[1] < self.bottom_right.y)

    def at_edge(self,other,edge_size): 
        '''
        Check if the other rectangle is at the edge of this one

        Params
        other     : Rectangle : Other rectangle to check
        edge_size : float     : width of edge 
        '''
        if (self.top_left.x     - edge_size) < other.top_left.x     < (self.top_left.x     + edge_size): 
            return True
        if (self.top_left.y     - edge_size) < other.top_left.y     < (self.top_left.y     + edge_size): 
            return True
        if (self.bottom_right.x - edge_size) < other.bottom_right.x < (self.bottom_right.x + edge_size): 
            return True 
        if (self.bottom_right.y - edge_size) < other.bottom_right.y < (self.bottom_right.y + edge_size): 
            return True
        return False
    
    def get_height_and_width(self):
        #Returns (height,width)
        width = self.bottom_right.x - self.top_left.x 
        height = self.bottom_right.y - self.top_left.y
        return (height,width)
    
    def bigger_rectangle(self,other): 
        '''
        Generate a bigger rectangle that encompass both this recangle and the provided rectangle

        Params
        other : Rectangle : other rectangle that resulting rectangle should encompass
        '''
        if self.top_left.x < other.top_left.x:
            x1 = self.top_left.x 
        else: 
            x1 = other.top_left.x 
        
        if self.top_left.y < other.top_left.y:
            y1 = self.top_left.y 
        else: 
            y1 = other.top_left.y 
        
        if self.bottom_right.x > other.bottom_right.x:
            x2 = self.bottom_right.x 
        else: 
            x2 = other.bottom_right.x
        
        if self.bottom_right.y > other.bottom_right.y:
            y2 = self.bottom_right.y 
        else: 
            y2 = other.bottom_right.y

        return Rectangle([x1,y1,x2,y2])

    def new_origo(self,new_origo):
        '''
        Create a new Rectangle object with the same dimensions but a new origo 

        Params
        new_origo : Point : Point describing new origo 
        '''
        x1 = self.top_left.x - new_origo.x 
        y1 = self.top_left.y - new_origo.y 
        x2 = self.bottom_right.x - new_origo.x 
        y2 = self.bottom_right.y - new_origo.y 
        
        return Rectangle([x1,y1,x2,y2])
    
    def get_area(self): 
        #Get area of rectangle 
        width,height = self.get_height_and_width()
        return width*height

    def draw(self,img,colour): 
        '''
        Draw rectangle in image
        
        Params
        img    : cv2 image    : image to draw rectangle in
        colour : tuple of int : 8 bit Color (b,g,r) 
        '''
        a = (int(self.top_left.x),int(self.top_left.y))
        b = (int(self.bottom_right.x),int(self.bottom_right.y))
        cv2.rectangle(img,a,b,colour,1)
    
    def __repr__(self):
        return("Rectangle([{x1},{y1},{x2},{y2})]".format(x1=self.top_left.x,y1=self.top_left.y,x2=self.bottom_right.x,y2=self.bottom_right.y))

