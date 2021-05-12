# This file contains classes for importing raw image files

import os
import numpy as np
import pandas as pd
import cv2
import copy 
import aicspylibczi
import pathlib
import xml.etree.ElementTree as ET

from coco_package import image_processing

VERBOSE = False

def set_verbose(): 
    global VERBOSE
    VERBOSE = True
    image_processing.set_verbose()

class Image_evos: 
    def __init__(self,z_planes): 
        '''
        Class to process evos files
        
        Params
        z_planes : list of str : paths to EVOS z_planes 
        '''

        self.z_planes = z_planes
        self.projection = None
        
        if VERBOSE: print("\tInitializing Image_evos. "+str(self))
        
    def make_min_projection(self):
        '''
        Convert z-stack of images to min-projection so we can export a Channel object instead of z_stack
        '''
        
        if VERBOSE: print("\tImage_evos: Making minimal projection")
        
        min_projection = None
        for path in self.z_planes: 
            plane = cv2.imread(path,cv2.IMREAD_ANYDEPTH)
            if min_projection is None:
                min_projection = plane
            else: 
               min_projection = np.minimum(min_projection,plane)
               
        max_pixel = min_projection.max()
        min_projection = (min_projection/(max_pixel/255)).astype('uint8')
        
        self.projection = min_projection
        
    def write_projection(self,path):
        '''
        Write projection to file
        
        Params
        path : str : path to write projection to 
        '''
        if not cv2.imwrite(path,self.projection):
            raise ValueError("Could not write projection to: "+path)
            
    def find_edges(self):
        """
        Find edges of obects in a stack of z_planes and save the projection of those edges
        """
        if VERBOSE: print("\tImage_evos: Finding edges")
        
        edges = None
        for plane_path in self.z_planes:
            edge = cv2.imread(plane_path,cv2.IMREAD_ANYDEPTH)
            max_pixel = np.amax(edge)
            edge = (edge/(max_pixel/255)).astype('uint8')
            edge = cv2.Canny(edge,100,200)
            edge = image_processing.Channel.remove_small_objects(edge,60)
            if edges is None: 
                edges = edge
            else:
                edges = edges + edge
                
        if edges is None: 
            raise ValueError("Could not properly read images. edges = "+str(edges)+", z_planes: "+str(self.z_planes))
        
        kernel_3 = np.ones((3,3),np.uint8)   
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_3)
        self.edges = edges     

    def get_channel(self,projection_path,categories):
        '''
        Return a Channel object 
        
        Params
        projection_path : str        : Path to write minimal projection to
        categories      : Categories : object describing categories

        Returns
        channel : Channel : object describing one channel to process
        '''
        self.make_min_projection()
        self.write_projection(projection_path)
        channel = image_processing.Channel(projection_path,channel_index = 0,z_index = 0,color = (255,255,255),categories = categories)
        self.find_edges()
        channel.mask = self.edges
        return channel
    
    @classmethod
    def walk_to_df(self,folder,id_split='-',filter_str=""):
        '''
        Find all files in folder recursively and report back as data frame with columns 
        
        Params
        folder      : str   : path to folder to look for files
        id_split    : str   : "id" is determined as everything before the first occurence of "id_split" in the file name. 
        filter_str  :       : do not include file names that have this string in their name
        
        Returns
        df          : pandas.DataFrame : data frame with information of all files in folder. "id" = file id, "root" = path to root folder, "file" = file name. 
        '''
        df = pd.DataFrame(columns = ["root","file"])
        f_id = []
        for root, dirs, fnames in os.walk(folder):
            if root is not list: 
                r = root
                root = [r]
            if len(fnames) > 0:
                for r in root: 
                    temp = pd.DataFrame(columns = ["root","file"])
                    temp["file"] = fnames
                    temp["root"] = r
                    temp = temp[temp["file"].str.contains(filter_str)]
                    df = df.append(temp,ignore_index=True)
        
        if id_split is not None:
            for f in df["file"]:
                name = f.split(id_split,1)[0]
                f_id.append(name)
        else:
            f_id = range(0,len(df["file"]))
        
        if not len(set(f_id)) == len(f_id):
            raise ValueError("Ids can't be equal")

        df["id"] = f_id
        df.set_index('id',inplace=True)
        
        if VERBOSE: print("Found "+str(len(df.index))+" files in folder: "+folder)
        
        return(df)
    
    @classmethod
    def evos_to_imageid(self,name):
        """ 
        Takes in EVOS image name and returns imageid 

        Params 
        name    : str : <exp>_<plate>_<day>_*_0_<well>f00d*.TIF

        Returns 
        imageid : str : <exp>_<plate>_<well>_<day>
        """
        if ".TIF" in name or ".tif" in name: 
            exp = name.split("_",3)[:3]
            well = name.split("_0_",1)[1].split("f00d",1)[0]
            return (exp[0]+"_"+exp[1]+"_"+well+"_"+exp[2])
        else: 
            return (None)
            
    @classmethod
    def find_stacks(self,folder):
        '''
        Find all stacks in folder and report them back as data frame with id column
        
        Params 
        folder : str : path to folder with all stacks
        
        Returns
        stacks : pandas.DataFrame : Data frame with path to stack and individual id of stack
        
        '''
        stacks = Image_evos.walk_to_df(folder,id_split=None,filter_str="TIF")
        if len(stacks) == 0:
            raise ValueError("Did not find any stacks")

        stacks_id = []
        stacks_path = []
        for i in stacks.index:
            stacks_id.append(Image_evos.evos_to_imageid(stacks.loc[i,"file"]))
            stacks_path.append(os.path.join(stacks.loc[i,"root"],stacks.loc[i,"file"]))
        stacks["id"] = stacks_id
        stacks["full_path"] = stacks_path
        
        return stacks

    def __repr__(self):
        string = "Image_evos: n_z_planes = {nz}".format(nz = len(self.z_planes))
        return string

class Image_czi:

    def __init__(self,raw_path,extracted_folder):
        '''
        Class to process czi files
        
        Params
        raw_path         : str : path to raw czi file to process
        extracted_folder : str : path to put extracted files
        '''
        
        self.raw_path = raw_path 
        self.file_id = os.path.splitext(os.path.split(self.raw_path)[1])[0]
        
        self.extracted_folder = os.path.join(extracted_folder,self.file_id)
        os.makedirs(self.extracted_folder,exist_ok=True)
        
        self.extracted_images_info_file = os.path.join(self.extracted_folder,"files_info.txt")
        
        self.OME_FILE_ENDING = ".ome.tiff"
        self.file_ending = copy.copy(self.OME_FILE_ENDING)
    
    def get_extracted_files_path(self,extract_method,max_projection = True):
        '''
        Extract images from microscopy image file. 
        
        Params
        extract_method : str  : Dependency to use for extracting images. one of "aicspylibczi", "bfconvert" and "imagej". 
        max_projection : bool : If aicspylibczi is selected, a maximal projection can be made in the extraction process
        
        Returns
        images_path : list of str : Path to extracted images. File names according to following convention: {file_id}_INFO_{series_index}_{time_index}_{z_index}_{channel_index}{file_ending}
        '''
        if extract_method == "aicspylibczi":
            self.file_ending = ".tiff"
        
        if not os.path.exists(self.extracted_images_info_file):
            if extract_method == "aicspylibczi":
                self.extract_aicspylibczi(max_projection)
            elif extract_method == "bfconvert": 
                if max_projection: 
                    raise ValueError("Max projection option not implemented for bfconvert extraction method")
                self.extract_images_bfconvert()
            elif extract_method == "imagej": 
                if max_projection: 
                    raise ValueError("Max projection option not implemented for imagej extraction method")
                self.extract_images_imagej()
            else: 
                raise ValueError("Not a valid extraction method. extract_method = "+extract_method)
    
        with open(self.extracted_images_info_file,'r') as f: 
            images_paths = f.read().splitlines()
        
        return images_paths
    
    def get_z_stack(self,segment_settings,categories,extract_method,max_projection = True,): 
        '''
        Convert a list of extracted images to Zstack classes
        
        Params
        segment_settings : list of Segment_settings : List of segment settings classes, one for each channel index
        categories       : Categories               : Categories relevant for this set of images
        
        Returns
        z_stacks    : list of Zstack : Files organized into Zstack classes for further use
        
        '''
        img_paths = self.get_extracted_files_path(extract_method,max_projection)
        
        df_images = pd.DataFrame(data = {"full_path":img_paths})
        df_images["root"],df_images["fname"] = df_images["full_path"].str.rsplit("/",1).str
        df_images["fid"],df_images["info"] = df_images["fname"].str.split("_INFO_",1).str
        df_images["info"] = df_images["info"].str.replace(self.file_ending,"")
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
                    found_segment_settings = False
                    for j in segment_settings: 
                        if str(j.channel_index) == str(this_image.loc[i,"channel_index"]):
                            color = j.color
                            global_max = j.global_max
                            found_segment_settings = True
                            break
                            
                    if not found_segment_settings: 
                        raise ValueError("Did not find segment settings for channel "+str(i))
                    
                    channel = image_processing.Channel(this_image.loc[i,"full_path"],this_image.loc[i,"channel_index"],this_image.loc[i,"z_index"],color,categories,global_max)
                    channels.append(channel)
                images.append(image_processing.Image(channels,z_index,segment_settings))
            
            file_id = this_z["fid"].iloc[0]
            time_index = this_z["time_index"].iloc[0]
            series_index = this_z["series_index"].iloc[0]
            
            z_stacks.append(image_processing.Zstack(images,file_id,series_index,time_index))
        
        for z in z_stacks: 
            if extract_method == "aicspylibczi":
                z.set_physical_res(self.resolution_aicspylibczi())
            else: 
                z.find_physical_res()
            
        return z_stacks
    
    def resolution_aicspylibczi(self): 
        '''
        Find resolution in meta_data file from extraction of images with aicspylibczi
        
        Returns 
        physical_size_x : dict : dictionary of resolutions of "x","y","z" and the "unit".  
        '''
        def line_to_value(line): 
            tag,value = line.split("\t")
            value = value.strip()
            return value
            
        metadata_path = os.path.join(self.extracted_folder,"meta_data.txt")
        
        physical_size = dict()
        with open(metadata_path,'r') as f: 
            lines = f.readlines()

        for l in lines: 
            if "ScalingX" in l: 
                physical_size['x'] = float(line_to_value(l))
            if "ScalingY" in l:     
                physical_size['y'] = float(line_to_value(l))
            if "ScalingZ" in l: 
                physical_size['z'] = float(line_to_value(l))
            if "guessed_unit" in l: 
                physical_size["unit"] = line_to_value(l)
        
        if physical_size["unit"] == "m":
            for k in physical_size.keys(): 
                physical_size[k] = physical_size[k] * 10**6
            physical_size["unit"] = "um"
        
        if VERBOSE: print("Read physical size from:"+metadata_path+"\n\t"+str(physical_size))
        
        return physical_size
    
    def recursive_xml_find(self,root,tag,new_tag_key=None): 
        '''
        Return first element in xml tree that have tag. Recursive.

        root        : lxml.etree : xml tree
        tag         : str        : tag of element to find
        new_tag_key : str        : 
        
        Returns
        result : list of str : List of elemnents in the form f"{tag}\t{value}"
        '''
        data = (list(root.iter(tag)))
        
        if new_tag_key is not None: 
            tag = new_tag_key
        
        result = []
        for d in data: 
            result.append(str(tag)+"\t"+str(d.text))
        
        return result

    def meta_xml_to_file(self,czi,meta_data_important_path,meta_data_all_path=None): 
        '''
        Extract some useful metadata from czi image object and write it to a file
        '''
        meta_data = czi.meta
        data = list()
        data = data + self.recursive_xml_find(meta_data,"ScalingX")
        data = data + self.recursive_xml_find(meta_data,"ScalingY")
        data = data + self.recursive_xml_find(meta_data,"ScalingZ")
        
        if len(data)==3:
            data = data + ["guessed_unit\tm"]
        else: 
            data = ["ScalingX\t1.0","ScalingY\t1.0","ScalingZ\t1.0","guessed_unit\tpx"]
            
            if VERBOSE: print("Warning! Did not find scaling of images. Setting to 1 px.")
        
        with open(meta_data_important_path,'w') as f: 
            for d in data: 
                f.write(d+"\n")
        
        if meta_data_all_path is not None: 
            other_data = []
            for elem in meta_data.iter():
                if elem.text is not None:
                    elem_text = str(elem.text)
                    if len(elem_text)>0:
                        other_data.append(str(elem.tag)+"\t"+elem_text)
            other_data = list(set(other_data))
            
            with open(meta_data_all_path,'w') as f: 
                for d in other_data: 
                    f.write(d+"\n")
            
    def extract_aicspylibczi(self,max_projection=True): 
        '''
        Use aicspylibczi to extract images from czi file. Assumes time and series dimensions = 1. 

        Params
        max_projection : bool : Make minimal projection of z_stack
        '''
        
        if VERBOSE: print("Reading file: " + self.raw_path + "\t and extracting to folder: " + self.extracted_folder)
        
        czi = aicspylibczi.CziFile(pathlib.Path(self.raw_path))
        dims = czi.dims_shape()[0]
        channels = list(range(dims['C'][1]))
        z_indexes = list(range(dims['Z'][1]))
        
        for c in channels: 
            img_projection = None 
            for z in z_indexes: 
                img = czi.read_mosaic(C=c,Z=z,scale_factor=1)
                img = np.squeeze(img)
                if not max_projection: 
                    out_path = out_path.join(self.extracted_folder,self.file_id + "_INFO_0_0_"+str(z)+"_"+str(c)+self.file_ending)
                    if VERBOSE: print("\tChannel: "+str(c)+"/"+str(max(channels))+"\tZ_index: "+str(z)+"/"+str(max(z))+"\tWriting: "+out_path)
                    if not cv2.imwrite(out_path,img):
                        raise ValueError("Could not save image to: "+out_path)
                else: 
                    if img_projection is None: 
                        img_projection = img
                    else: 
                        img_projection = np.maximum(img_projection,img)
            if max_projection:
                out_path = os.path.join(self.extracted_folder,self.file_id + "_INFO_0_0_0_"+str(c)+self.file_ending)
                if VERBOSE: print("\tChannel: "+str(c)+"/"+str(max(channels))+"\tWriting: "+out_path)
                if not cv2.imwrite(out_path,img_projection):
                    raise ValueError("Could not save image to: "+out_path)
                
        self.meta_xml_to_file(czi,os.path.join(self.extracted_folder,"meta_data.txt"),os.path.join(self.extracted_folder,"meta_data_all.txt"))
        
        img_paths = []
        for path in os.listdir(self.extracted_folder):
            if path.endswith(".tiff"):
                img_paths.append(os.path.join(self.extracted_folder,path))
        
        with open(self.extracted_images_info_file,'w') as f: 
            for path in img_paths: 
                f.write(path+"\n")
    
    def extract_bfconvert(self):
        '''
        Use bfconvert tool to get individual stack images from microscopy format file
        
        Legacy function. Only aicspylibczi is currently supported
        '''
        
        bfconvert_info_str = "_INFO_%s_%t_%z_%c"
        
        out_name = self.file_id+bfconvert_info_str+self.file_ending 
        out_name = os.path.join(self.extracted_folder,out_name)
        log_file = os.path.join(os.path.split(self.extracted_folder)[0],"log_"+self.file_id+".txt")

        cmd = "bfconvert -overwrite \"{file_path}\" \"{out_name}\" > {log_file}".format(file_path = self.raw_path,out_name = out_name,log_file=log_file)
        
        if VERBOSE: 
            print(cmd)
        exit_val = os.system(cmd)
        if exit_val !=0:
            raise RuntimeError("This command did not exit properly: \n"+cmd)
        
        img_paths = [] 
        
        for path in os.listdir(self.extracted_folder):
            if path.endswith(self.file_ending):
                img_paths.append(os.path.join(self.extracted_folder,path))
        
        with open(os.path.join(self.extracted_folder,self.extracted_images_info_file),'w') as f: 
            for path in img_paths: 
                f.write(path+"\n")
        

    def extract_imagej(self,use_xvfb = True):
        '''
        Use imagej to get individual stack images from microscopy format file. This script also stitch together images in x-y plane.
        
        Sometimes you get problems when ImageJ tries to reuse the same session. I think this might be solved by unchecking the box "Run single instance listener" under Edit --> Options --> Misc...  
        
        How to enable auto-stitch of czi files in ImageJ:
        1. Open FIJI2. Navigate to Plugins > Bio-Formats > Bio-Formats Plugins Configuration
        2. Select Formats
        3. Select your desired file format (e.g. “Zeiss CZI”) and select “Windowless”

        Legacy function. Only aicspylibczi is currently supported
        
        Params 
        use_xvfb : bool : Run imagej with xvfb-run command to avoid need for showing imagej bar etc.
        '''
        
        info_split_str = "_INFO"

        this_script_folder = os.path.split(os.path.abspath(__file__))[0]
        imagej_macro_path = os.path.join(this_script_folder,"stitch_w_imagej.ijm")
        
        raw_path_full = os.path.abspath(self.raw_path)
        fname = os.path.splitext(os.path.split(self.raw_path)[1])[0]
        out_name = fname#+info_split_str+self.file_ending 
        out_name = os.path.abspath(os.path.join(self.extracted_folder,out_name))
        log_file = os.path.abspath(os.path.join(os.path.split(self.extracted_folder)[0],"log_"+fname+".txt"))
        
        cmd = "ImageJ-linux64 --ij2 --console -macro \"{imagej_macro_path}\" \"{file_path},{out_name}\"".format(imagej_macro_path = imagej_macro_path,file_path = raw_path_full,out_name = out_name,log_file = log_file)
        if use_xvfb: 
            cmd = "xvfb-run -a "+cmd
        
        if VERBOSE:
            print(cmd)
        
        bash_script = os.path.join(self.extracted_folder,"temp.sh")
        with open(bash_script,"w") as f: 
            f.write("#!/bin/sh\n")
            f.write(cmd)

        os.system("chmod u+x "+bash_script)

        exit_val = os.system(bash_script)
        
        print("command exit_val = "+str(exit_val))
        
        if exit_val !=0:
            print("NB! imagej command did not finish properly!")
            #raise RuntimeError("This command did not exit properly: \n"+cmd)
        
        img_paths = os.listdir(self.extracted_folder)
        for old_path in img_paths: 
            if old_path.endswith(self.file_ending):
                file_id,zid_and_channel = old_path.split("_INFO_",1)
                zid_and_channel = zid_and_channel.split("-",4)
                z_id = zid_and_channel[1]
                channel = zid_and_channel[3]
                new_path = file_id + "_INFO_0_0_"+z_id+"_"+channel+".ome.tiff"
                os.rename(os.path.join(self.extracted_folder,old_path),os.path.join(self.extracted_folder,new_path))

        img_paths = []
        for path in os.listdir(self.extracted_folder):
            if path.endswith(self.file_ending):
                img_paths.append(os.path.join(self.extracted_folder,path))
        
        with open(self.extracted_images_info_file,'w') as f: 
            for path in img_paths: 
                f.write(path+"\n")
        
    
    def __repr__(self):
        string = "Img_file: raw_path = {r},extracted_folder = {e}".format(r = self.raw_path,e=self.extracted_folder)
    
    def __lt__(self,other):
        return self.file_id < other.file_id
        