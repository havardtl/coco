import os
import numpy as np
import pandas as pd
import cv2

MASKS_IN_MEMORY = True #Store channel masks in memory instead of in written files
IMGS_IN_MEMORY = True #Store channel images in memory instead of in written files
MIN_CONTOUR_AREA = 5

VERBOSE = False 

TEMP_FOLDER = None
CONTOURS_STATS_FOLDER = None

contour_id_counter = 0
roi3d_id_counter = 0

class Segment_settings: 

    def __init__(self,channel_index,shrink,contrast,auto_max,thresh_type, thresh_upper, thresh_lower,open_kernel,close_kernel,combine):
        '''
        Object for storing segmentation settings

        Params
        channel_index   : int    : Channel index of channnel these settings account for
        shrink          : float  : Value from 0 to 1 to shrink mask by
        contrast        : float  : Increase contrast by this value. Multiplied by image. 
        auto_max        : bool   : Autolevel mask before segmenting
        thresh_type     : str    : Type of threshold to apply. Either ["canny_edge","binary"]
        thresh_upper    : int    : Upper threshold value
        thresh_lower    : int    : Lower threshold value
        open_kernel     : int    : Size of open kernel 
        close_kernel    : int    : Size of closing kernel 
        combine         : bool   : Whether or not this channel should be combined into combined value 
        '''
        global VERBOSE,TEMP_FOLDER,CONTOURS_STATS_FOLDER 
        
        self.AVAILABLE_THRESH_TYPES = ["canny_edge","binary"] 
        
        self.channel_index = self.to_int_or_none(channel_index)
        self.shrink = self.to_int_or_none(shrink)
        self.contrast = self.to_int_or_none(contrast)
        self.auto_max = self.to_bool_or_none(auto_max)
        
        assert thresh_type in self.AVAILABLE_THRESH_TYPES,"Chosen thresh type not in AVAILABLE_THRESH_TYPES: "+str(self.AVAILABLE_THRESH_TYPES)
        self.thresh_type = thresh_type
        
        thresh_upper = self.to_int_or_none(thresh_upper)
        thresh_lower = self.to_int_or_none(thresh_lower)
        
        if thresh_upper is not None: 
            assert 0<thresh_upper<256,"thresh_upper not in range (0,255)"
        
        if thresh_lower is not None: 
            assert 0<thresh_lower<256,"thresh_lower not in range (0,255)"

        if self.thresh_type == "binary": 
            assert thresh_upper == 255,"Upper threshold must be 255 if binary threshold"

        self.thresh_upper = thresh_upper 
        self.thresh_lower = thresh_lower
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


    def __repr__(self):
        string = "{class_str}: channel = {ch}, shrink = {s},contrast = {c},auto_max = {a},thresh_type = {tt},thresh_upper = {tu}, thresh_lower = {tl},open_kernel = {ok}, close_kernel = {ck}".format(class_str = self.__class__.__name__,ch = self.channel_index,s=self.shrink,c=self.contrast,a=self.auto_max,tt=self.thresh_type,tu=self.thresh_upper,tl=self.thresh_lower,ok = self.open_kernel,ck = self.close_kernel)
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
        img_number   : str  : Identifier string for image in the well
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
        
        self.img_id = str(self.experiment)+"_"+str(self.plate)+"_"+str(self.time)+"_"+str(self.well)+"_"+str(self.other_info)
        self.id_z_stack = self.img_id+"_S"+str(self.series_index)+"_T"+str(self.time_index)
        
        self.mask_save_folder = os.path.join(TEMP_FOLDER,"masks",self.id_z_stack)
        os.makedirs(self.mask_save_folder,exist_ok = True)
        self.xml_folder = os.path.join(TEMP_FOLDER,"xml_files",self.id_z_stack)
        os.makedirs(self.xml_folder,exist_ok = True)
        self.combined_mask_folder = os.path.join(TEMP_FOLDER,"combined_masks",self.id_z_stack)
        os.makedirs(self.combined_mask_folder,exist_ok=True)

        self.img_dim = self.images[0].get_img_dim()
        self.x_res,self.y_res,self.z_res = self.get_physical_res()
        
        if VERBOSE:
            print("Made z_stack: "+str(self))

    def make_masks(self):
        #Generate masks in all Images objects
        if VERBOSE: 
            print("Making masks for all images in z_stack "+str(self))
        for i in self.images:
            i.make_masks(self.mask_save_folder)

    def find_contours(self): 
        #Find contours in all Images objects
        if VERBOSE: 
            print("Finding contours all images in z_stack "+str(self))
        for i in self.images: 
            i.find_contours()

    def find_z_overlapping(self): 
        #Find all overlapping contours for all contours in images
        if VERBOSE: 
            print("Finding z_overlapping for all images in z_stack "+str(self))
        for i in range(len(self.images)-1):
            self.images[i].find_z_overlapping(self.images[i+1])
    
    def update_contour_stats(self):
        if VERBOSE: 
            print("Updating contour stats for all contours in z_stack "+str(self))
        for i in self.images: 
            for j in i.channels:
                for k in j.contours: 
                    k.update_contour_stats()

    def get_rois_3d(self):
        '''
        Get rois 3d from all contour objects in all images

        Returns 
        rois_3d : list of Roi_3d : All Roi_3d objects 
        '''
        if VERBOSE: 
            print("Generating roi_3ds for all images in z_stack "+str(self))
        rois_3d = []
        for i in self.images:
            i.make_rois_3d(rois_3d,self) 
        
        return rois_3d

    def get_physical_res(self):
        #Get physical resolution for the z_stack in um per pixel
        return self.images[0].get_physical_res(self.xml_folder)
    
    def is_inside_combined(self):
        #Make combined channel and then check whether the contour is inside for all images
        if VERBOSE: 
            print("Making combined channels for all images in z_stack "+str(self))
        for i in self.images:
            i.make_combined_channel(self.img_id,self.combined_mask_folder)
        
        if VERBOSE: 
            print("Finding if all contours are inside combined mask for all images in z_stack "+str(self))
        for i in self.images:
            i.is_inside_combined() 
   
    def measure_channels(self):
        #Measure contours for all channels
        for i in self.images:
            i.measure_channels()

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
        
        if VERBOSE:
            print("Made Image object: "+str(self))
        
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
                    combined_mask = c.get_mask() 
                else: 
                    combined_mask = cv2.bitwise_and(combined_mask,c.get_mask())

        combined_mask_path = os.path.join(combined_mask_folder,z_stack_name+self.id_image+".png")
        cv2.imwrite(combined_mask_path,combined_mask)
        
        self.combined_mask = Channel(combined_mask_path,-1,self.z_index)
        self.combined_mask.mask = combined_mask
        self.combined_mask.find_contours()
        
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

    def find_contours(self): 
        #Make contour objects for all channels in image. Requires channels to have a generated mask. 
        for i in self.channels: 
            i.find_contours()
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
        return None 

    def make_rois_3d(self,rois_3d,z_stack):
        ''' 
        Make Roi_3d objects and add them to the variable roi_3d 

        Params
        rois_3d : list of Rois_3D : List to add generated Roi_3d objects to
        z_stack : Z_stack         : Z_stack of which the generated roi_3d belongs to 
        '''
        for i in self.channels: 
            i.make_rois_3d(rois_3d,z_stack,self)
        return None 

    def get_img_dim(self):
        #Get image dimensions 
        img = self.channels[0].get_image()
        return img.shape 

    def get_physical_res(self,xml_folder):
        #Get physical resoluion of image in um per pixel 
        return self.channels[0].get_physical_res(xml_folder)
       
    def measure_channels(self):
        #Measure the mean grey and sum positive pixels for each contour in channels for all channels
        for i in self.channels: 
            for j in self.channels:
                for k in i.contours:
                    k.measure_channel(j)
        return None
    
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
    def __init__(self,full_path,channel_index,z_index):
        '''
        
        '''
        global VERBOSE 
        self.full_path = full_path 
        self.root_path,self.file_name = os.path.split(full_path)
        self.file_id = os.path.splitext(self.file_name)[0]
        self.channel_index = int(channel_index)
        self.id_channel = str(channel_index)
        self.z_index = z_index
        
        self.image = None 
        self.mask = None
        self.mask_path = None
        self.contours = None 
        self.n_contours = None
        
        if VERBOSE: 
            print("Made channel: "+str(self))

    def get_image(self):
        if self.image is not None: 
            return self.image 
        else: 
            image = cv2.imread(self.full_path,cv2.IMREAD_ANYDEPTH)
            
            global IMGS_IN_MEMORY 
            if IMGS_IN_MEMORY: 
                self.image = image 
            
            return image
    
    def get_mask(self):
        if self.mask is not None: 
            return self.mask
        elif self.mask_path is not None: 
            mask = cv2.imread(self.mask_path,cv2.IMREAD_GRAYSCALE)
            return mask
        else: 
            raise RuntimeError("Need to create mask before it can be returned")

    def make_mask(self,segment_settings,mask_save_folder):
        '''
        Generate a mask image of the channel where 255 == true pixel, 0 == false pixel 

        Params
        segment_settings : Segment_settings : Object containing all the segment settings
        mask_save_folder : str              : where to save the output folder
        
        '''
        
        if self.mask_path is not None: 
            if os.path.isfile(self.mask_path): 
                return None

        img = self.get_image()
        assert img is not None, "Can't make mask out of Image == None"
        
        s = segment_settings 
        
        if (s.shrink !=1) and (s.shrink is not None):
            new_dim = (int(img.shape[0]*s.shrink),int(img.shape[1]*s.shrink))
            img = cv2.resize(img,new_dim,interpolation = cv2.INTER_AREA) 
        
        if (s.contrast !=1) and s.contrast is not None:
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
        
        assert img is not None, "Mask is none after running function. That should not happen"

        global MASKS_IN_MEMORY 
        if MASKS_IN_MEMORY: 
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

    def make_rois_3d(self,rois_3d,z_stack,image):
        ''' 
        Make Roi_3d objects and add them to the variable roi_3d 

        Params
        rois_3d : list of Rois_3D : List to add generated Roi_3d objects to
        z_stack : Z_stack         : Z_stack of which the generated roi_3d belongs to 
        image   : Image           : Image of which the generated roi_3d belongs to
        '''
        for i in self.contours: 
            i.make_rois_3d(rois_3d,z_stack,image,self)
    
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
        assert self.contours is not None,"Need to run Channel.find_contours() before contours can be evaluated"
        assert other_channel.contours is not None,"Need to run Channel.find_contours() before contours can be evaluated"
        for i in self.contours: 
            for j in other_channel.contours:
                i.is_inside(j)
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
        only_contour = np.zeros(self.img_dim,dtype="uint8")
        cv2.drawContours(only_contour,[self.points],-1,color=255,thickness = -1)
        self.only_contour = only_contour 

        self.data = None#self.contour_stats() 
        
        #if VERBOSE:
        #    print("Made contour: "+str(self))
        
    def is_at_edge(self):
        '''
        Check if this contour is overlapping with the edge of the image
        '''
        at_edge = (np.sum(self.only_contour[0,:])> 0) or (np.sum(self.only_contour[:,0]) > 0)or (np.sum(self.only_contour[:,self.img_dim[0]-1])>0) or (np.sum(self.only_contour[self.img_dim[1]-1,:])>0)
        return at_edge 
    
    def measure_channel(self,channel):
        mean_grey =  cv2.bitwise_and(channel.get_image(),channel.get_image(),mask=self.only_contour)
        mean_grey = np.sum(mean_grey)
        sum_pos_pixels =  cv2.bitwise_and(channel.get_mask(),channel.get_mask(),mask=self.only_contour)
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
            "z_overlapping":self.z_overlapping_as_str(), 
            "is_inside":self.is_inside 
        }
        
        self.data = data 
        return None        

    def is_inside(self,other_contour):
        '''
        Check if contour is inside other contour
        
        Params
        other_contour : Contour : Contour to check if it is inside 
        '''
        overlap_img = cv2.bitwise_and(other_contour.only_contour,other_contour.only_contour,mask = self.only_contour)
        if self.is_inside is None: 
            self.is_inside = []
        if overlap_img.sum()>0:
            self.is_inside.append(other_contour)
        return None

    def find_z_overlapping(self,next_z): 
        '''
        Check if this contour overlaps with any contour in the other channel

        Params
        next_z : Channel : The next z-plane of the same channels 

        '''
        empty_img = np.zeros(self.img_dim,dtype="uint8")
        this_contour = empty_img.copy()
        cv2.drawContours(this_contour,[self.points],-1,color=255,thickness = -1)
        if self.z_overlapping is not None: 
            raise ValueError("Contour.z_overlapping is already set. Did you run Contour.find_z_overlapping() twice ?")
        
        self.z_overlapping = []

        for next_z_c in next_z.contours: 
            next_contour = empty_img.copy()
            cv2.drawContours(next_contour,[next_z_c.points],-1,color=255,thickness=-1)

            overlap_img = cv2.bitwise_and(next_contour,next_contour,mask = this_contour)
            if overlap_img.sum() > 5:
                    self.z_overlapping.append(next_z_c)
                    next_z_c.overlapps = True

        return None
    
    def make_rois_3d(self,rois_3d,z_stack,image,channel):
        ''' 
        Make Roi_3d objects and add them to the variable roi_3d 

        Params
        rois_3d : list of Rois_3D : List to add generated Roi_3d objects to
        z_stack : Z_stack         : Z_stack of which the generated roi_3d belongs to 
        image   : Image           : Image of which the generated roi_3d belongs to
        channel : Channel         : Channel of which the generated roi_3d belongs to 
        '''
        if self.z_overlapping is None: 
            raise RuntimeError("You have to find z_overlapping contours before making rois_3d")
        
        if not self.overlapps:
            all_z_overlapping = []
            all_z_overlapping = self.get_all_z_overlapping(all_z_overlapping)
            #for z in all_z_overlapping: 
            #    print("\t"+str(z))
            new_roi = Roi_3d(all_z_overlapping,z_stack,image,channel)
            rois_3d.append(new_roi)

        return None

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

    def z_overlapping_as_str(self):
        if self.z_overlapping is None:
            return None
        else: 
            z_overlapping_str = "" 
            for z in self.z_overlapping: 
                if z_overlapping_str is "": 
                    z_overlapping_str = str(z.id_contour) 
                else: 
                    z_overlapping_str = z_overlapping_str + "," + str(z.id_contour)
            return z_overlapping_str 

    def print_all(self):
        print("\t\t\t",end="")
        print(self)

    def __repr__(self):
        string = "{class_str}, contour_index: {class_id}, n points: {n}, z_overlapping: {z}".format(class_str = self.__class__.__name__,class_id = self.id_contour,n = len(self.points),z=self.z_overlapping_as_str())
        return string

class Roi_3d: 

    def __init__(self,contours,z_stack,image,channel):
        self.contours = contours
        self.z_stack = z_stack
        self.image = image
        self.channel = channel 

        self.data = None

        global roi3d_id_counter 
        self.id_roi_3d = roi3d_id_counter 
        roi3d_id_counter = roi3d_id_counter + 1

    def build(self):
        '''
        Build data about roi_3d object from the list of contours and other information
        '''

        data = {
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
            "image_id"     : self.image.id_image,
            "channel_id"   : self.channel.id_channel,
            "is_inside"    : None,
            "volume"       : 0,
            "contour_ids"  : None,
            "contours_centers_xyz" : None,
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
            
            if data["at_edge"] is None:
                    data["at_edge"] = c.data["at_edge"] 
            elif c.data["at_edge"] is not None:
                if c.data["at_edge"]: 
                    data["at_edge"] = c.data["at_edge"]
            
            if data["is_inside"] is None:
                data["is_inside"] = str(c.data["is_inside"])
            else:
                data["is_inside"] = data["is_inside"] +","+str(c.data["is_inside"])
        
        self.data = data

    def __repr__(self):
        string = "{class_str} id: {class_id} built from n contours: {n}".format(class_str = self.__class__.__.name__,class_id = self.id_roi_3d,n = len(self.contours))
        return string
    
