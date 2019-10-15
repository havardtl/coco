import os
import math
import numpy as np
import pandas as pd
import cv2

IMGS_IN_MEMORY = True #Store channel images in memory instead of in written files
MIN_CONTOUR_AREA = 5

VERBOSE = False 

TEMP_FOLDER = "coco_temp"
CONTOURS_STATS_FOLDER = "contour_stats"
GRAPHICAL_SEGMENTATION_FOLDER = "segmented_graphical"
PROJECTIONS_RAW_FOLDER = "projections_raw"

THIS_SCRIPT_FOLDER = os.path.split(os.path.abspath(__file__))[0]

DEFAULT_SETTINGS_XLSX = os.path.join(THIS_SCRIPT_FOLDER,"default_annotations.xlsx")
TEST_SETTINGS_SHEET = "test_settings"
SEGMENT_SETTINGS_SHEET = "segment_settings"
ANNOTATION_SHEET = "annotation"
PLOT_VARS_SHEET = "plot_vars"

contour_id_counter = 0
roi3d_id_counter = 0
contour_chain_counter = 0

class Segment_settings: 

    def __init__(self,channel_index,color,contrast,auto_max,thresh_type, thresh_upper, thresh_lower,open_kernel,close_kernel,combine):
        '''
        Object for storing segmentation settings

        Params
        channel_index   : int           : Channel index of channnel these settings account for
        color           : str           : Color to convert channel into if needed in 8-bit BGR format. "B,G,R" where 0 <= b,g,r <= 255
        contrast        : float         : Increase contrast by this value. Multiplied by image. 
        auto_max        : bool          : Autolevel mask before segmenting
        thresh_type     : str           : Type of threshold to apply. Either ["canny_edge","binary"]
        thresh_upper    : int           : Upper threshold value
        thresh_lower    : int           : Lower threshold value
        open_kernel     : int           : Size of open kernel 
        close_kernel    : int           : Size of closing kernel 
        combine         : bool          : Whether or not this channel should be combined into combined value 
        '''
        global VERBOSE,TEMP_FOLDER,CONTOURS_STATS_FOLDER 
        
        self.AVAILABLE_THRESH_TYPES = ["canny_edge","binary"] 
        
        self.channel_index = self.to_int_or_none(channel_index)
        
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
        if size is not None:
            size = int(size)
            kernel = np.ones((size,size),np.uint8)
        else: 
            kernel = None 
        return kernel
     
    def to_float_or_none(self,value): 
        # If value is not none, turn it into a int
        if value is not None: 
            return float(value)
        else: 
            return None
    
    def to_int_or_none(self,value): 
        # If value is not none, turn it into a int
        if value is not None: 
            return int(value)
        else: 
            return None
    
    def to_bool_or_none(self,value): 
        # If value is not none, turn it into a bool
        if value is not None: 
            return bool(value)
        else: 
            return None
    
    def get_dict(self):
        data = {
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

    def __repr__(self):
        string = "{class_str}: channel = {ch}, color = {col},contrast = {c},auto_max = {a},thresh_type = {tt},thresh_upper = {tu}, thresh_lower = {tl},open_kernel = {ok}, close_kernel = {ck}".format(class_str = self.__class__.__name__,ch = self.channel_index,col=self.color,c=self.contrast,a=self.auto_max,tt=self.thresh_type,tu=self.thresh_upper,tl=self.thresh_lower,ok = self.open_kernel_int,ck = self.close_kernel_int)
        return string

class Zstack: 

    def __init__(self,images,experiment,plate,time,well,img_numb,other_info,series_index,time_index):
        '''
        Object representing a z_stack
        
        Params
        images       : list : List of Images objects
        experiment   : str  : Identifier string for experiment 
        plate        : str  : Identifier string for plate 
        time         : str  : Identifier string for day 
        well         : str  : Identifier string for well
        img_numb     : str  : Identifier string for image in the well
        other_info   : str  : Other info about z_stack
        series_index : str  : Index of image series when imaging multiple location in one go 
        time_index   : str  : Index of time index in timed experiments
        '''
        
        global VERBOSE,TEMP_FOLDER,CONTOURS_STATS_FOLDER
        
        assert type(images) is list,"images must be a list of image objects"
        for i in images: 
            assert isinstance(i,Image),"found an image in images that is not an image object"
        self.images = images
        self.images.sort()
        
        self.experiment = experiment
        self.plate = plate
        self.time = time 
        self.well = well
        self.img_numb = img_numb
        self.other_info = other_info

        self.series_index = series_index 
        self.time_index = time_index 
        
        self.img_id = str(self.experiment)+"_"+str(self.plate)+"_"+str(self.time)+"_"+str(self.well)+"_"+str(self.img_numb)+"_"+str(self.other_info)
        self.id_z_stack = self.img_id+"_S"+str(self.series_index)+"_T"+str(self.time_index)
        
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
        self.x_res,self.y_res,self.z_res = (None,None,None)
        self.find_physical_res()

        self.projections = None
        self.composite = None

        self.made_combined_masks = False
        
    def make_masks(self):
        #Generate masks in all Images objects
        if VERBOSE: print("Making masks for all images in z_stack "+str(self))
        
        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            i.make_masks(self.mask_save_folder)
            
        if VERBOSE: print("")
        
    def make_combined_masks(self):
        #Combine all masks into one to make a super object that other contours can be checked if they are inside
        if VERBOSE: print("Making combined channels for all images in z_stack "+str(self))

        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            i.make_combined_channel(self.id_z_stack,self.combined_mask_folder)
        
        if VERBOSE: print("")
        self.made_combined_masks = True 

    def find_contours(self): 
        #Find contours in all Images objects
        if VERBOSE: print("Finding contours all images in z_stack "+str(self))
        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            for j in i.channels: 
                j.find_contours()
            
            if self.made_combined_masks:
                i.combined_mask.find_contours()
        if VERBOSE: print("")

    def find_z_overlapping(self): 
        #Find all overlapping contours for all contours in images
        if VERBOSE: print("Finding z_overlapping for all images in z_stack "+str(self))
        for i in range(len(self.images)-1):
            if VERBOSE: print(i,end="  ",flush=True)
            self.images[i].find_z_overlapping(self.images[i+1])
        
        last_image = self.images[len(self.images)-1]
        last_channels = last_image.channels
        if self.made_combined_masks: 
            last_channels = last_channels + [last_image.combined_mask]
        for i in last_channels: 
            for j in i.contours: 
                j.z_overlapping = []
                
        if VERBOSE: print("")
            
    def update_contour_stats(self):
        if VERBOSE: print("Updating contour stats for all contours in z_stack "+str(self))
        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            all_channels = i.channels
            if self.made_combined_masks: 
                all_channels = all_channels + [i.combined_mask]
            for j in all_channels:
                for k in j.contours: 
                    k.update_contour_stats()
                    
        if VERBOSE: print("")
        
    def get_rois_3d(self):
        '''
        Get rois 3d from all contour objects in all images

        Returns 
        rois_3d : list of Roi_3d : All Roi_3d objects 
        '''
        if VERBOSE: print("Generating roi_3ds for all images in z_stack "+str(self))

        for i in self.images: 
            for k in i.combined_mask.contours: 
                k.add_roi_id()
            for j in i.channels: 
                for k in j.contours: 
                    k.add_roi_id()
        
        combined_rois = []
        for i in self.images:
            for k in i.combined_mask.contours:
                k.add_to_roi_list(combined_rois,self,i.combined_mask.channel_index)
        
        rois_3d = []
        for i in self.images: 
            for j in i.channels: 
                channel_index = j.channel_index
                for k in j.contours: 
                    k.add_to_roi_list(rois_3d,self,channel_index)
        
        for i in rois_3d:
            i.is_inside_combined_roi(combined_rois)
        
        rois_3d = combined_rois + rois_3d

        return rois_3d 
    
    def find_physical_res(self):
        #Get physical resolution for the z_stack in um per pixel
        self.x_res,self.y_res,self.z_res = self.images[0].get_physical_res(self.xml_folder)
        for i in self.images: 
            for j in i.channels: 
                j.x_res = self.x_res
    
    def is_inside_combined(self):
        #Check whether the contour is inside for all images
        if VERBOSE: 
            print("Finding if all contours are inside combined mask for all images in z_stack "+str(self))
        for i in self.images:
            if VERBOSE: print(i.z_index,end="  ",flush=True)
            i.is_inside_combined() 
        if VERBOSE: print("")
        
    def measure_channels(self):
        #Measure contours for all channels
        for i in self.images:
            all_channels = i.channels
            
            if self.made_combined_masks: 
                all_channels = i.channels + [i.combined_mask]
            
            for j in all_channels: 
                for k in j.contours:
                    k.measure_channel(i.channels)

    def write_contour_info(self):
        #Write out all information about contours
        df = pd.DataFrame() 
        for i in self.images: 
            z_index = i.z_index 
            for j in i.channels:
                channel_id = j.id_channel 
                for k in j.contours:
                    contour_id = k.id_contour 
                    temp = pd.DataFrame(k.data) 
                    temp["z_index"] = z_index 
                    temp["channel_id"] = channel_id
                    temp["contour_id"] = contour_id
                    df = pd.concat([df,temp],ignore_index = True,sort=False)
        
        df["z_stack_id"] = self.id_z_stack 
        df["x_res_um"] = self.x_res
        df["y_res_um"] = self.y_res
        df["z_res_um"] = self.z_res 
        df["experiment"] = self.experiment 
        df["plate"] = self.plate 
        df["time"] = self.time  
        df["well"] = self.well 
        df["img_numb"] = self.img_numb 
        df["other_info"] = self.other_info 
        df["series_index"]  = self.series_index  
        df["time_index"] = self.time_index  

        contours_stats_path = os.path.join(CONTOURS_STATS_FOLDER,self.id_z_stack+".csv")
        if VERBOSE:
            print("Writing info about contours to: "+contours_stats_path)
        df.to_csv(contours_stats_path)
    
    def make_projections(self,max_projection = True,auto_max=True,colorize = True,add_scale_bar = True):
        '''
        Make projections from z_planes by finding the minimal or maximal value in the z-axis
        Params 
        z_planes        : list : a list of paths to each plane in a stack of z_planes
        max_projection  : bool : Use maximal projection, false gives minimal projection
        auto_max        : bool : Auto level image after making projection
        colorize        : bool : Add color to image from segment_settings
        add_scale_bar   : bool : Add scale bar to image
        
        Updates
        self.projections : list of np.array : List of the projections of the z_planes as an 8-bit image. Ordered by channel_index.
        self.composite   : np.array         : Composite image of the projections
        '''
        if VERBOSE: print("Making projections for z_stack: "+self.id_z_stack)
        
        global PROJECTIONS_RAW_FOLDER
        
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
                    new_channel_path = os.path.join(PROJECTIONS_RAW_FOLDER,self.id_z_stack+"_C"+str(j.channel_index)+".png")
                    new_channel = Channel(new_channel_path,j.channel_index,None,j.color)
                    new_channel.image = j.get_image().copy()
                    projections.append(new_channel)
        
        max_channel_index = 0
        for p in projections: 
            cv2.imwrite(p.full_path,p.get_image())
        
            p.x_res = self.x_res
            if p.channel_index > max_channel_index: 
                max_channel_index = p.channel_index
        
        composite_img = None
        for p in projections: 
            p_modified = p.make_img_for_viewing(PROJECTIONS_RAW_FOLDER,scale_bar=add_scale_bar,auto_max=auto_max,colorize=colorize)
            if composite_img is None: 
                composite_img = p_modified
            else: 
                composite_img = cv2.add(composite_img,p_modified) 
        
        self.projections = projections
        
        new_channel_path = os.path.join(PROJECTIONS_RAW_FOLDER,self.id_z_stack+"_Ccomp"+".png")
        composite = Channel(new_channel_path,max_channel_index + 1,None,None)
        composite.image = composite_img
                
        composite.make_img_for_viewing(PROJECTIONS_RAW_FOLDER,scale_bar=False,auto_max=False,colorize=False)
        
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
        
        df["file_id"] = self.img_id
        df["id_z_stack"] = self.id_z_stack 
        #df["experiment"] = self.experiment
        #df["plate"] = self.plate
        #df["time"] = self.time
        #df["well"] = self.well
        #df["img_numb"] = self.img_numb
        #df["other_info"] = self.other_info
        df["series_index"] = self.series_index
        df["time_index"] = self.time_index

        df["img_dim"] = str(self.img_dim) 
        
        return df
    
    def to_pdf(self,add_scale_bar = True,auto_max=True,colorize=True):
        '''
        Convert z_stack into a pdf that displays information about how each channel was segmented

        Params
        add_scale_bar : bool : Add a scale bar to image before plotting PDF
        auto_max      : bool : Scale pixels in image so that highest brightness pixel equals max pixel value
        colorize      : bool : add channel color to image  
        '''
        if VERBOSE:
            print("Making PDF from z_stack "+self.id_z_stack) 
        df = pd.DataFrame() 
        
        pdf_imgs = []
        x = 0
        y = 0
        x_vars = ["channel_id","type","auto_max"]
        y_vars = ["z_index"]
        image_vars = ["file_id"]
        data = None
        for i in self.images: 
            z_index = i.z_index
            for j in i.channels:
                channel_id = j.id_channel 
                file_id = j.id_channel
                data = {"z_index":i.z_index,"file_id":j.file_id,"channel_id":j.id_channel,"type":None,"auto_max":auto_max}

                if add_scale_bar or auto_max or colorize: 
                    j.make_img_for_viewing(self.img_for_viewing_folder,add_scale_bar,auto_max,colorize)
                    full_image = j.img_for_viewing_path 
                else: 
                    full_image = j.full_path

                data["type"] = "original"
                data["auto_max"] = None 
                pdf_imgs.append(Image_in_pdf(x,y,full_image,data.copy(),x_vars,y_vars,image_vars))
                x = x + 1
                
                '''
                data["type"] = "mask"
                data["auto_max"] = None 
                mask_path = j.get_mask_as_file(self.mask_save_folder)
                x = x + 1
                pdf_imgs.append(Image_in_pdf(x,y,mask_path,data.copy(),x_vars,y_vars,image_vars))
                '''

                data["type"] = "w_contours"
                data["auto_max"] = False
                j.make_img_with_contours(self.img_w_contour_folder,colorize)
                pdf_imgs.append(Image_in_pdf(x,y,j.img_w_contour_path,data.copy(),x_vars,y_vars,image_vars))
                x = x + 1
            
            data["file_id"] = i.combined_mask.file_id
            data["type"] = "combined_mask"
            data["auto_max"] = None
            data["channel_id"] = -1
            pdf_imgs.append(Image_in_pdf(x,y,i.combined_mask.mask_path,data.copy(),x_vars,y_vars,image_vars))
            x = x + 1
            
            data["type"] = "w_contours"
            data["auto_max"] = None
            data["channel_id"] = -1
            i.combined_mask.make_img_with_contours(self.img_w_contour_folder,colorize=False)
            pdf_imgs.append(Image_in_pdf(x,y,i.combined_mask.img_w_contour_path,data.copy(),x_vars,y_vars,image_vars))
            
            y = y + 1
            x = 0

        save_path = os.path.join(GRAPHICAL_SEGMENTATION_FOLDER,self.id_z_stack+".pdf")
        pdf = Pdf(save_path,pdf_imgs,self.img_dim)
        pdf.make_pdf()

        return None

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
        channels         : list of Channel  : List containing each individual channel for this image
        z_index          : int              : z_index of the image
        segment_settings : Segment_settings : Object containing segmentation information for making masks 
        '''
        global VERBOSE 

        channels.sort() 
        self.channels = channels 
        self.z_index = int(z_index) 
        self.segment_settings = segment_settings
        self.id_image = str(z_index)

        self.combined_mask = None 
        
    def make_combined_channel(self,z_stack_name,combined_mask_folder):
        '''
        Merge masks into a channel. Useful for determining objects that other structures are inside. 
        Merges channels that are marked to merge in segment_settings. 
        
        Params
        z_stack_name         : str : Name of z_stack 
        combined_mask_folder : str : Folder to save combined mask in 
        empty_img_path       : str : path to all black image in same dimensions as combined masks
        '''
        channel_index = -1

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
        
        combined_mask_path = os.path.join(combined_mask_folder,z_stack_name+"_z"+self.id_image+".png")
        cv2.imwrite(combined_mask_path,combined_mask)
        
        empty_img = np.zeros(combined_mask.shape,dtype="uint8")
        empty_img_path = os.path.join(combined_mask_folder,z_stack_name+"_"+str(self.z_index)+"_"+str(channel_index)+".png")
        cv2.imwrite(empty_img_path,empty_img)
        
        self.combined_mask = Channel(empty_img_path,channel_index,self.z_index,None)
        self.combined_mask.mask = combined_mask
        self.combined_mask.mask_path = combined_mask_path
        return None

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
        img = self.channels[0].get_image()
        return img.shape 

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
        string = "{class_str} z_index: {class_id} with n channels: {n}".format(class_str = self.__class__.__name__,class_id = self.id_image,n = len(self.channels))
        return string

class Channel: 
    def __init__(self,full_path,channel_index,z_index,color):
        '''
        Object that represents a single channel in an image. 

        Params
        full_path     : str           : The whole path to image 
        channel_index : int           : index of channels. Must start at 0 for first channel and iterate +1
        z_index       : int           : z_index of current channel 
        color         : (int,int,int) : 8-bit color of current channel in BGR format
        '''
        global VERBOSE 
        self.full_path = full_path 
        self.root_path,self.file_name = os.path.split(full_path)
        self.file_id = os.path.splitext(self.file_name)[0]
        if self.file_id.endswith(".ome"):
            self.file_id = self.file_id[:-4]

        self.channel_index = int(channel_index)
        self.id_channel = str(channel_index)
        self.z_index = z_index

        self.x_res = None
        self.color = color
        self.img_for_viewing_path = None  #Add scalebar and colorize etc for viewing
        
        self.image = None 
        self.mask = None
        self.mask_path = None
        self.contours = None 
        self.n_contours = None

        self.img_w_contour_path = None
        
    def get_image(self):
        if self.image is not None: 
            return self.image 
        else: 
            image = cv2.imread(self.full_path,cv2.IMREAD_GRAYSCALE)
            
            global IMGS_IN_MEMORY 
            if IMGS_IN_MEMORY: 
                self.image = image 
            
            return image
     
    def make_img_for_viewing(self,img_for_viewing_folder,scale_bar=False,auto_max=False,colorize=False):
        '''
        Add image modified for viewing to channel

        Params
        img_for_viewing_folder : str  : Path to folder to save image in
        scale_bar              : bool : Add scale bar to image 
        auto_max               : bool : Make the highest intensity pixel the highest in image
        colorize               : bool : Convert grayscale image to color as specified in channel color
        
        Returns
        img                    : np.array : cv2 image after processing.
        '''
        img = self.get_image()
        self.img_for_viewing_path = os.path.join(img_for_viewing_folder,self.file_id+".png")
        
        if not auto_max and not scale_bar and not colorize: 
            cv2.imwrite(self.img_for_viewing_path,img)
            return img
        
        if auto_max:
            max_pixel = np.max(np.amax(img,axis=0))+1
            img = (img/(max_pixel/255)).astype('uint8')
        
        if colorize:
            img = self.gray_to_color(img,self.color)

        if scale_bar: 
            if self.x_res is None: 
                raise RuntimeError("Channel.x_res must be set outside class initiation to increase performance. Try Channel.get_physical_res()")
            img = self.add_scale_bar_to_img(img,self.x_res)
        
        cv2.imwrite(self.img_for_viewing_path,img)
        
        return img
    
    def make_img_with_contours(self,img_w_contour_folder,colorize):
        '''
        Make a version of the image with contours and the roi_3d_id if that one exists

        Params
        img_w_contour_folder : str  : Path to folder to save resulting image in 
        colorize             : bool : Convert from grayscale to channel specific color
        '''
        img = self.get_image().copy()
        
        if colorize:
            img = self.gray_to_color(img,self.color)
        else: 
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        contours = []
        for i in self.contours: 
            cv2.drawContours(img,[i.points],-1,(255,255,255),1)
            if i.roi_3d_id is not None:
                x = i.data["centroid_x"]
                y = i.data["centroid_y"]
                cv2.putText(img,str(i.roi_3d_id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
        
        img_w_contour_path = os.path.join(img_w_contour_folder,self.file_id+".png")
        cv2.imwrite(img_w_contour_path,img)
        
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
    def add_scale_bar_to_img(self,image,x_res):
        '''
        Add scale bar to image
        
        Params
        image : np.array : cv2 image to add scale bar to
        x_res : float    : Number of um each pixel in x-direction corresponds to
        '''
        image = image.copy()
        shape = image.shape 
        img_width = shape[1]
        img_height = shape[0]

        scale_in_pixels = int(img_width * 0.2)
        scale_in_um = scale_in_pixels * x_res
        
        unit = "um"
        scale_in_um_str = ("{:.0f} {unit}").format(scale_in_um,unit=unit)
        
        scale_height = int(scale_in_pixels * 0.05)
        margin = img_width*0.05
        x1 = int(img_width - (scale_in_pixels+margin))
        y1 = int(img_height*0.9)
        scale_box = [x1,y1,x1+scale_in_pixels,y1+scale_height] #[x1,y1,x2,y2] with x0,y0 in left bottom corner

        cv2.rectangle(image,(scale_box[0],scale_box[1]),(scale_box[2],scale_box[3]),(255,255,255),thickness = -1)

        text_x = scale_box[0] + int(img_width*0.02)
        text_y = scale_box[1] - int(img_height*0.02)
        
        font_size = img_width/1000  
        
        cv2.putText(image,scale_in_um_str,(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,font_size,(255,255,255),2)

        return image
   
    def get_mask_as_file(self,mask_save_folder): 
        if self.mask_path is not None: 
            return self.mask_path
        else: 
            self.mask_path = os.path.join(mask_save_folder,str(self.file_id)+".png")
            cv2.imwrite(self.mask_path,self.mask)
            return self.mask_path

    def get_mask(self):
        if self.mask is not None: 
            return self.mask
        elif self.mask_path is not None: 
            mask = cv2.imread(self.mask_path,cv2.IMREAD_GRAYSCALE)
            return mask
        else: 
            raise RuntimeError("Need to create mask before it can be returned")

    def make_mask(self,segment_settings,mask_save_folder = None):
        '''
        Generate a mask image of the channel where 255 == true pixel, 0 == false pixel 

        Params
        segment_settings : Segment_settings : Object containing all the segment settings
        mask_save_folder : str              : where to save the output folder. If none, masks are forced to be saved in memory
        
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

            if type(s.close_kernel) is np.ndarray:  
                img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,s.close_kernel)
            
            if type(s.open_kernel) is np.ndarray:  
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, s.open_kernel)
        
        assert img is not None, "Mask is none after running function. That should not happen"
        
        if mask_save_folder is None: 
            mask_in_memory = True
        else: 
            global IMGS_IN_MEMORY
            mask_in_memory = IMGS_IN_MEMORY
        
        if mask_in_memory: 
            self.mask = img
        else: 
            self.mask_path = os.path.join(mask_save_folder,str(self.file_id)+".png")
            cv2.imwrite(self.mask_path,img)
        
        return None 
    
    def find_contours(self): 
        '''
        Find contours in masked image of channel 
        '''
        mask = self.get_mask()
        contours_raw, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        global MIN_CONTOUR_AREA
        
        contours = []
        img_dim = mask.shape
        for i in range(len(contours_raw)):
            M = cv2.moments(contours_raw[i])
            contour_area = M['m00']
            if contour_area > MIN_CONTOUR_AREA:
                contours.append(Contour(contours_raw[i],i,img_dim,self.z_index))
        
        self.contours = contours
        self.n_contours = len(self.contours) 
        return None

    def find_z_overlapping(self,next_z):
        '''
        Find all contours in next z plane in same channel that overlaps with this object

        Params
        next_z : Channel : image channel with contours to compare against
        '''

        for i in self.contours: 
            i.find_z_overlapping(next_z)
        

    def print_all(self):
        print("\t\t",end="")
        print(self)
        if self.contours is not None: 
            for i in self.contours: 
                i.print_all()

    def get_physical_res(self,xml_folder):
        '''
        Find physical resolution from ome.tiff file

        Returns
        physical_size_x : float : x resolution of each pixel in um
        physical_size_y : float : y resolution of each pixel in um
        physical_size_z : float : z resolution of each pixel in um
        '''
        if "ome.tiff" not in self.file_name: 
            raise ValueError("Can only find xyz resolution from ome.tiff files, not from file: "+ome_tiff_path)
        
        xml_path = os.path.join(xml_folder,self.file_id+".xml")

        cmd = "tiffcomment {image} > {xml}".format(image = self.full_path,xml = xml_path)
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

        supported_units = ["µm"]
        
        if unit_x not in supported_units:
            raise ValueError("unit_x = "+unit_x+" is not a supported unit. Supported units: "+str(supported_units))
        if unit_y not in supported_units:
            raise ValueError("unit_y = "+unit_y+" is not a supported unit. Supported units: "+str(supported_units))
        if unit_z not in supported_units:
            raise ValueError("unit_x = "+unit_z+" is not a supported unit. Supported units: "+str(supported_units))
        
        if VERBOSE:
            unit_str = "x: {x} {x_u}\ty: {y} {y_u} \t z: {z} {z_u}".format(x=physical_size_x,x_u=unit_x,y=physical_size_y,y_u=unit_y,z=physical_size_z,z_u=unit_z)
            print("Found physical res equal "+unit_str+" for "+str(self))

        return physical_size_x,physical_size_y,physical_size_z
    
    def is_inside_combined(self,other_channel):
        '''
        For all contours in channel, find those contours which it is in other_channel 
        
        Params
        other_channel : Channel : Channel object which contours to check if contours in this channel is inside 
        '''
        assert self.contours is not None,"Must run Channel.find_contours() before contours can be evaluated"
        assert other_channel.contours is not None,"Must to run Channel.find_contours() before contours can be evaluated"
        for i in self.contours:
            for j in other_channel.contours:
                i.is_inside_other_contour(j)
        return None

    def __lt__(self,other):
        return self.channel_index < other.channel_index 
    
    def __repr__(self):
        string = "{class_str} channel_index: {class_id} with n contours: {n}".format(class_str = self.__class__.__name__,class_id = self.id_channel,n = self.n_contours)
        return string

class Contour:

    def __init__(self,points,contour_index,img_dim,z_index):
        global VERBOSE 
        self.points = points 
        self.contour_index = int(contour_index)
        self.img_dim = img_dim 

        global contour_id_counter
        self.id_contour = contour_id_counter 
        contour_id_counter = contour_id_counter + 1
        
        self.z_index = z_index
        self.z_overlapping = None 
        self.overlapps = False #True if this Contour is stored in another Contours z_overlapping value
        
        self.is_inside = None
        
        self.contour_box = self.get_contour_rectangle()
        
        self.roi_3d_id = None #Id of this chain of rois that make up a single Roi_3d

        self.data = None#self.contour_stats() 
    
    def get_contour_rectangle(self):
        '''
        Calculate the bounding box around the contour and return the coordinates as a rectangle object
        
        Returns
        rectangle : Rectangle : Rectangle object that contains the contour  
        '''
        x,y,w,h = cv2.boundingRect(self.points)
        
        box = [x,y,x+w,y+h]
        
        return Rectangle(box)
    
    def get_only_contour(self):
        '''
        Draw a black image with only contour.

        Returns 
        only_contour : np.array : cv2 image with only this contour
        '''
        only_contour = np.zeros(self.img_dim,dtype="uint8")
        cv2.drawContours(only_contour,[self.points],-1,color=255,thickness = -1)
        return only_contour
    
    def is_at_edge(self):
        '''
        Check if this contour is overlapping with the edge of the image
        '''
        only_contour = self.get_only_contour()
        
        bw = 1
        img_dim = only_contour.shape[:2]
        mask = np.ones(img_dim, dtype = "uint8")
        cv2.rectangle(mask, (bw,bw),(img_dim[1]-bw,img_dim[0]-bw), 0, -1)
        only_edge = cv2.bitwise_and(only_contour, only_contour, mask = mask)
        
        if np.sum(only_edge) > 0 : 
            return True
        else: 
            return False
    
    def measure_channel(self,channels):
        '''
        Measure this contours sum grey and sum positive pixels for all channels

        Params
        channels : list of Channel : channels to measure
        '''
        only_contour = self.get_only_contour()
        
        for channel in channels:
            mean_grey =  cv2.bitwise_and(channel.get_image(),channel.get_image(),mask=only_contour)
            mean_grey = np.sum(mean_grey)
            sum_pos_pixels =  cv2.bitwise_and(channel.get_mask(),channel.get_mask(),mask=only_contour)
            sum_pos_pixels = np.sum(sum_pos_pixels/255)
            
            data = {
                "sum_grey_C"+str(channel.channel_index):[mean_grey],
                "sum_positive_C"+str(channel.channel_index):[sum_pos_pixels]}
            
            self.data.update(data)

        return None   

    def update_contour_stats(self):
        M = cv2.moments(self.points)
        area = M['m00']
        perimeter = cv2.arcLength(self.points,True)
        (x_circle,y_circle),radius = cv2.minEnclosingCircle(self.points)
        
        data = {
            "img_dim":str(self.img_dim),
            "area":area,
            "centroid_x":int(M['m10']/M['m00']),
            "centroid_y":int(M['m01']/M['m00']),
            "z_index":self.z_index,
            "perimeter":perimeter,
            "hull_tot_area":cv2.contourArea(cv2.convexHull(self.points)),
            "radius_enclosing_circle":radius,
            "equvialent_diameter":np.sqrt(4*area/np.pi),
            "circularity":float(4)*np.pi*area/(perimeter*perimeter),
            "at_edge":self.is_at_edge(),
            "z_overlapping":self.contour_list_as_str(self.z_overlapping), 
            "is_inside":self.contour_list_as_str(self.is_inside) 
        }
        
        self.data = data 
        return None        
            
    def is_inside_other_contour(self,other_contour):
        '''
        Check if contour is inside other contour
        
        Params
        other_contour : Contour : Contour to check if it is inside 
        '''
        if self.is_inside is None: 
            self.is_inside = []
        
        overlaps = self.contour_box.overlaps(other_contour.contour_box)
        
        if overlaps: 
            only_this_contour = self.get_only_contour()
            only_other_contour = other_contour.get_only_contour()
            overlap_img = cv2.bitwise_and(only_other_contour,only_other_contour,mask = only_this_contour)
            if overlap_img.sum()>0:
                self.is_inside.append(other_contour)

    def find_z_overlapping(self,next_z): 
        '''
        Check if this contour overlaps with any contour in the other channel

        Params
        next_z : Channel : The next z-plane of the same channels 

        '''
        if self.z_overlapping is not None: 
            raise ValueError("Contour.z_overlapping is already set. Did you run Contour.find_z_overlapping() twice ?")
        
        only_this_contour = self.get_only_contour()
        self.z_overlapping = []
        for next_z_c in next_z.contours: 
            overlaps = self.contour_box.overlaps(next_z_c.contour_box)
            
            if overlaps: 
                only_other_contour = other_contour.get_only_contour()
                overlap_img = cv2.bitwise_and(only_other_contour,only_other_contour,mask = only_this_contour)
                if overlap_img.sum()>5:
                    self.z_overlapping.append(next_z_c)
                    next_z_c.overlapps = True
    
    def add_roi_id(self):
        all_z_overlapping = []
        all_z_overlapping = self.get_all_z_overlapping(all_z_overlapping)
        
        this_roi_id = None 
        for i in all_z_overlapping:
            if i.roi_3d_id is not None:
                if this_roi_id is not None: 
                    assert this_roi_id == i.roi_3d_id,"Finding conflicting roi_3d_id for overlapping contour"
                this_roi_id = i.roi_3d_id 
        
        if this_roi_id is None: 
            global roi3d_id_counter
            this_roi_id = roi3d_id_counter
            roi3d_id_counter = roi3d_id_counter + 1

        for i in all_z_overlapping: 
            i.roi_3d_id = this_roi_id 

    def get_all_z_overlapping(self,all_z_overlapping):
        '''
        Look through the contours recursively and return all contours that are overlapping in z

        Params
        all_z_overlapping : list of Contour : List of all contours that are overlapping in z_dimension 
        '''
        all_z_overlapping.append(self)
        for z in self.z_overlapping:
            z.get_all_z_overlapping(all_z_overlapping)
        
        return all_z_overlapping
    
    def add_to_roi_list(self,rois_3d,z_stack,channel_index):
        ''' 
        If this contour has a roi_3d_id similar to an exisiting Roi_3d object, add it to that, if not make a new Roi_3d with the contour 

        Params
        rois_3d           : list of Rois_3D : List of Rois_3d to add contour to
        z_stack           : Z_stack         : Z_stack of which the generated roi_3d belongs to 
        channel_index     : int             : Index of the channel of which the contour belongs to 
        '''
        make_new_roi_3d = True  
        for i in rois_3d: 
            if i.id_roi_3d == self.roi_3d_id: 
                make_new_roi_3d = False
                i.contours.append(self)

        if make_new_roi_3d: 
            new_roi = Roi_3d(self.roi_3d_id,[self],z_stack,channel_index)
            rois_3d.append(new_roi)
        
        return None

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
        string = "{class_str}, contour_index: {class_id}, n_points: {n}, roi_3d_id: {roi}".format(class_str = self.__class__.__name__,class_id = self.id_contour,n = len(self.points),roi = self.roi_3d_id)
        return string

class Roi_3d: 

    def __init__(self,id_roi_3d,contours,z_stack,channel_index):
        '''
        Region of interest in 3 dimensions. Built from 2D contours that overlapp.

        Params
        id_roi_3d     : int              : id of the roi_3d object
        contours      : list of Contours : Contours that specify 2D objects in image
        z_stack       : Z_stack          : z_stack these contours are from
        channel_index : int              : Channel index these contours are from
        '''
        self.id_roi_3d = id_roi_3d 
        self.contours = contours
        self.z_stack = z_stack
        self.channel_index = channel_index 

        self.is_inside_roi = None 

        self.data = None
    
    def is_inside_combined_roi(self,combined_rois):
        '''
        Each contour in a Roi_3d has information about which combined_mask contour it is inside. This function looks through Roi_3d objects and converts information about which contour it is inside to which combined_mask Roi_3d object it is inside. 
        
        Params
        combined_rois : list of Roi_3d : List of contours to check whether these contours are inside         
        '''
        is_inside_roi = []
        for c in self.contours:
            for c_i in c.is_inside:

                for cr in combined_rois:
                    for crc in cr.contours:
                        if c_i.id_contour == crc.id_contour:
                            is_inside_roi.append(cr.id_roi_3d)
        is_inside_roi = list(set(is_inside_roi))
        self.is_inside_roi = is_inside_roi

    def build(self):
        '''
        Build data about roi_3d object from the list of contours and other information
        '''

        data = {
            "id_roi_3d"    : self.id_roi_3d,
            "z_stack"      : self.z_stack.id_z_stack,
            "x_res_um"     : self.z_stack.x_res,
            "y_res_um"     : self.z_stack.y_res,
            "z_res_um"     : self.z_stack.z_res,
            "experiment"   : self.z_stack.experiment, 
            "plate"        : self.z_stack.plate, 
            "time"         : self.z_stack.time,  
            "well"         : self.z_stack.well, 
            "img_numb"     : self.z_stack.img_numb, 
            "other_info"   : self.z_stack.other_info, 
            "series_index" : self.z_stack.series_index,  
            "time_index"   : self.z_stack.time_index,
            "channel_index": self.channel_index,
            "is_inside"    : None,
            "is_inside_roi": None,
            "volume"       : 0,
            "n_contours"   : len(self.contours),
            "contour_ids"  : None,
            "contours_centers_xyz" : None,
            "mean_x_pos" : None,
            "mean_y_pos" : None,
            "mean_z_index" : None, 
            "at_edge"      : None
        }

        to_um = data["x_res_um"] * data["y_res_um"] * data["z_res_um"]

        mean_grey_channels = [s for s in self.contours[0].data.keys() if "sum_grey_C" in s]
        sum_pos_pixels_channels = [s for s in self.contours[0].data.keys() if "sum_positive_C" in s]
        
        temp = {} 
        for tag in mean_grey_channels+sum_pos_pixels_channels: 
            temp.update({tag:0})
         
        for c in self.contours: 
            for tag in mean_grey_channels+sum_pos_pixels_channels:
                temp[tag] = temp[tag] + (c.data[tag][0] *to_um)
        
        data.update(temp)
        if self.is_inside_roi is not None:  
            for inside_roi in self.is_inside_roi: 
                if data["is_inside_roi"] is None: 
                    data["is_inside_roi"] = str(inside_roi)
                else: 
                    data["is_inside_roi"] = data["is_inside_roi"] +"," +str(inside_roi)
        
        sum_x_pos = 0
        sum_y_pos = 0
        sum_z_pos = 0

        for c in self.contours:
            data["volume"] = data["volume"] + (c.data["area"]*to_um)
            
            if data["contour_ids"] is None: 
                data["contour_ids"] = str(c.id_contour)
            else:
                data["contour_ids"] = data["contour_ids"] +","+str(c.id_contour)
            
            center = "x"+str(c.data["centroid_x"])+"y"+str(c.data["centroid_y"])+"z"+str(c.data["z_index"])
            if data["contours_centers_xyz"] is None: 
                data["contours_centers_xyz"] = center 
            else:
                data["contours_centers_xyz"] = data["contours_centers_xyz"]+","+center
           
            sum_x_pos += int(c.data["centroid_x"])
            sum_y_pos += int(c.data["centroid_y"])
            sum_y_pos += int(c.data["z_index"])

            if data["at_edge"] is None:
                    data["at_edge"] = c.data["at_edge"] 
            elif c.data["at_edge"] is not None:
                if c.data["at_edge"]: 
                    data["at_edge"] = c.data["at_edge"]
            
            if data["is_inside"] is None:
                data["is_inside"] = str(c.data["is_inside"])
            else:
                data["is_inside"] = data["is_inside"] +","+str(c.data["is_inside"])
        
        if data["n_contours"]>0:
            data["mean_x_pos"]   = sum_x_pos/data["n_contours"]
            data["mean_y_pos"]   = sum_y_pos/data["n_contours"]
            data["mean_z_index"] = sum_z_pos/data["n_contours"]

        self.data = data

    def __repr__(self):
        string = "{class_str} id: {class_id} built from n contours: {n}".format(class_str = self.__class__.__name__,class_id = self.id_roi_3d,n = len(self.contours))
        return string

class Rectangle:
    def __init__(self,points):
        '''
        Rectangle object
        '''
        self.bottom_left  = [points[0],points[1]]
        self.top_right    = [points[2],points[3]]
    
    def overlaps(self,other):
        '''
        Check if this rectangle overlaps with another 
        
        Params
        other    : Rectangle : Other rectangle to check
        
        Returns 
        overlaps : bool      : True if rectangles overlap
        '''
        return not (self.top_right[0] < other.bottom_left[0] or self.bottom_left[0] > other.top_right[0] or self.top_right[1] < other.bottom_left[1] or self.bottom_left[1] > other.top_right[1])
    
class Image_in_pdf:
    
    def __init__(self,x,y,img_path,data,x_vars,y_vars,image_vars):
        '''
        Information about single image that is needed to make Pdf document
        
        Params
        x           : int         : Relative x position of image. zero-indexed
        y           : int         : Relative y position of image. zero-indexed
        data        : dict        : Dictionary with all the metadata about image to plot in annotation boxes around images
        x_vars      : x_vars      : Variables in "data" to plot on x-axis annotation fields 
        y_vars      : y_vars      : Variables in "data" to plot on y-axis annotation fields 
        Image_vars  : list of str : Variables in "data" to plot on top of image
        '''
        self.x = x
        self.y = y
        self.img_path = img_path
        self.data = data
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.image_vars = image_vars
    
    def __repr__(self):
        string = "{class_str}: xy: ({x},{y}), {data}".format(class_str = self.__class__.__name__,x = self.x,y = self.y, data = self.data)
        return string

class Pdf:
    
    def __init__(self,save_path,images_in_pdf,image_dim):
        '''
        Plot images in a pdf sorted by info about images
        
        Params
        save_path     : str                  : Path to save pdf in
        images_in_pdf : list of Image_in_pdf : List of objects with information about image path, location in pdf and x and y variables to plot
        image_dim     : tuple of int         : image format in pixels [height,width]. Adds grey border to images not in this aspect ratio 
        '''
        
        self.save_path = save_path
        self.images_in_pdf = images_in_pdf
        self.image_dim = image_dim
        
        self.check_overlapping_imgs()
        
    def check_overlapping_imgs(self): 
        all_xy = []
        for i in self.images_in_pdf: 
            all_xy.append("x"+str(i.x)+"y"+str(i.y))
        
        set_xy = set(all_xy)
         
        difference = len(all_xy) - len(set_xy)
        
        if difference > 0: 
            raise ValueError("The images_in_pdf list has overlapping values. all: {a}, set {s}, difference: {d}".format(a = len(all_xy),s = len(set_xy),d=difference))
        
    def shorten_info(self,info,box_length,char_marg=3):
        '''
        If info is larger than a certain length, shorten it

        Params
        info       : str   : Information to shorten
        box_length : float : Length of box to fit info into
        char_marg  : int   : Character margin to subtract from length
        '''
        short_info = info 
        
        avg_character_width = 3.5
        max_characters = int(box_length/avg_character_width) - char_marg

        if len(short_info)>max_characters:
            short_info = short_info.split(" = ")[1]
            if len(short_info)>max_characters:
                short_info = short_info[0:max_characters]
        #print("info: "+info+"\tlen: "+str(len(info))+"\tshorten_info: "+short_info+"\tlen: "+str(len(short_info)))
        return short_info 
    
    def draw_annotation_pdf(self,canvas,box_dims,vertical_text,text):
        '''
        Make an annotation box in your pdf document.  
        Params
        canvas     : reportlab.Canvas : pdf dokument to draw box in
        box_dims   : tuple of float   : Position of box (x1,y1,width,height)
        text_angle : float            : Angle to rotate text by 
        '''
        
        from reportlab.lib.colors import Color
        canvas.setFont("Helvetica",7)
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

    def make_pdf(self,annotate_box_height = 10):
        '''
        Make PDF as specified
        
        Params
        annotate_box_width : int : Height of annotation box in same direction as text
        '''
        
        if VERBOSE: 
            print("Saving PDF to: "+self.save_path)

        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.lib.utils import ImageReader 
        
        canvas = Canvas(self.save_path,pagesize = A4)

        c_width,c_height = A4

        y_max = 0
        x_max = 0
        for i in self.images_in_pdf:
            if i.x > x_max: 
                x_max = i.x
            if i.y > y_max: 
                y_max = i.y

        area_meta_yvars = (0,(len(self.images_in_pdf[0].y_vars))*annotate_box_height)
        area_meta_xvars = (0,(len(self.images_in_pdf[0].x_vars))*annotate_box_height)
        marg = 2
        goal_aspect_ratio = self.image_dim[0]/self.image_dim[1]

        img_width = (c_width-area_meta_yvars[1])/(x_max+1)
        img_height = img_width * goal_aspect_ratio

        if img_height > c_height: 
            img_height = c_height
        
        y_img_per_page = math.floor((c_height-area_meta_yvars[1])/img_height)
        
        yvars_box_width  = (area_meta_yvars[1]-area_meta_yvars[0])/(len(self.images_in_pdf[0].y_vars))
        xvars_box_height = (area_meta_xvars[1]-area_meta_yvars[0])/(len(self.images_in_pdf[0].x_vars))
        page = 0
        for img in self.images_in_pdf:
            image = ImageReader(img.img_path)
            
            x = (img.x*img_width) + area_meta_yvars[1]
            y_position = img.y - (page*y_img_per_page) + 1
            if y_position > y_img_per_page: 
                page = page +1 
                canvas.showPage()
                y_position = y_position- y_img_per_page
            y = c_height - ((y_position)*img_height)- area_meta_xvars[1]
            
            image_x = x+marg/2
            image_y = y+marg/2
            canvas.drawImage(image,image_x,image_y,width = img_width-marg, height = img_height-marg)
            
            image_info = None 
            for j in img.image_vars: 
                if image_info is None: 
                    image_info = img.data[j]
                else: 
                    image_info = image_info + "   " +str(img.data[j])
            
            canvas.setFont("Helvetica",2)
            canvas.setFillColorRGB(0.5,0.5,0.5)
            canvas.drawString(image_x+1,image_y+1,image_info)       
            
            canvas.saveState()
            
            last_info = ""
            for j in range(len(img.y_vars)): 
                info = img.y_vars[j]+" = "+str(img.data[img.y_vars[j]])
                if info != last_info:
                    x_ymeta = yvars_box_width*(j) 
                    short_info = self.shorten_info(info,img_height)
                    self.draw_annotation_pdf(canvas,[x_ymeta+marg/4,y+marg/2,yvars_box_width-marg/2,img_height-marg],True,short_info)
                    last_info = info
            
            last_info = ""
            for j in range(len(img.x_vars)): 
                info = img.x_vars[j]+" = "+str(img.data[img.x_vars[j]])
                    
                if info != last_info:
                    y_xmeta = xvars_box_height*(j+1) 
                    short_info = self.shorten_info(info,img_width)
                    self.draw_annotation_pdf(canvas,[x+marg/2,c_height-(y_xmeta+marg/4),img_width-marg,xvars_box_height-marg/4],False,short_info)
            canvas.restoreState()
        canvas.save()


