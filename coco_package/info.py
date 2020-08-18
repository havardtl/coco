# This file contains classes to organize metadata about images
import pandas as pd
import io
import os

VERBOSE = False

def set_verbose(): 
    VERBOSE = True

class Image_info:
    def __init__(self,id,img_path=None,annotation_path=None,manually_reviewed=None):
        '''
        Class describing image info
        
        Params
        id                : str  : Unique id-string for object
        img_path          : str  : Path to image file
        annotation_path   : str  : Path to annotation file for image
        manually_reviewed : bool : Whether this image has been manually reviewed
        '''
        self.id = id
        self.img_path = img_path
        self.annotation_path = annotation_path
        self.manually_reviewed = manually_reviewed
    
    def add_annotation_path(self,session): 
        '''
        Add path to annotation from session
        
        Params
        session : Session : instance of session 
        '''
        self.annotation_path = session.get_annotation_path(self.id)

    def add_img_path(self,session): 
        '''
        Add path to minimal projection from session
        
        Params
        session : Session : instance of session 
        '''
        self.img_path = session.get_img_path(self.id)    
        
    def __lt__(self,other):
        return self.id < other.id 
    
    
class Session: 
    def __init__(self,path,image_infos,index):
        '''
        Linking together annotation and the image it annotates
        
        Params
        path        : str                : path to Session file
        image_infos : list of Image_info : List of class with all necessary info about image
        index       : int                : position in list of image_info that is currently selected
        '''
        self.path = path
        self.image_infos = image_infos
        self.index = index
        
        self.n_images = len(self.image_infos)
        self.image_numbs = list(range(0,self.n_images))
        
        if len(self.image_infos)<1:
            raise ValueError("image_infos must be a list longer than 0. image_info: "+str(self.image_infos))

        if len(self.image_infos)<index+1:
            raise ValueError("index is bigger than the length of image_infos. len(image_infos): "+str(len(self.image_infos))+" index: "+str(index))
    
    @classmethod
    def from_file(self,path):
        '''
        Read session file from path
        
        Params
        path : str : path to session file. csv file with fields "id", "root_image", "file_image", "root_annotation", "file_annotation", "manually_reviewed"
        
        Returns
        session : Session : instance of session class 
        '''
        df = pd.read_csv(path)
        df.set_index("id",inplace=True)
        df["image_numb"] = list(range(0,len(df.index)))
        
        image_infos = []
        for i in df.index:
            img_path  = os.path.join(df.loc[i,"root_image"],df.loc[i,"file_image"])
            img_annot = os.path.join(df.loc[i,"root_annotation"],df.loc[i,"file_annotation"])
            reviewed = df.loc[i,"manually_reviewed"]
            image_infos.append(Image_info(i,img_path,img_annot,reviewed))
    
        index = 0
        for i in df.index:
            if not df.loc[i,'manually_reviewed']:
                index = df.loc[i,"image_numb"]
                break

        session = Session(path,image_infos,index)
        return session
    
    def sort(self):
        #Make image_infos appear in sorted order
        self.image_infos.sort()
    
    def select_first(self,n):
        '''
        Select the first n rows for processing
        
        Params
        n : int : selects 0:n first Image_info instances
        '''
        self.image_infos = self.image_infos[0:n]
    
    def find_missing(self,rawdata_ids): 
        '''
        Check if some ids that are found in session file are not found in the rawdata

        Params
        rawdata_ids : list of str : ids in raw data
        '''
        all_ids = []
        for i in self.image_infos:
            all_ids.append(i.id)
        
        missing_from_stacks = set(set(all_ids)).difference(rawdata_ids)
        if len(missing_from_stacks)>0:
            found_in_stacks = set(rawdata_ids).intersection(set(all_ids))
            raise ValueError("Rawdata missing? IDs from session file not found in stacks: "+str(missing_from_stacks)+"\n\nFound in stacks: "+str(found_in_stacks))
    
    def get_n(self):
        #Return the number of images in session 
        return self.n_images
    
    def reset_index(self):
        #Set index to start
        self.index = 0
    
    def get_process_info(self):
        '''
        Get information relevant to image currently being processed
        
        process_info : str : {image_numb}/{tot_images}\t{index}
        '''
        return "{image_numb}/{tot_images}\t{index}".format(image_numb = self.index+1,tot_images = self.n_images,index = self.image_infos[self.index].id)
        
    def get_img_info(self):
        '''
        Get currently selected Image_info instance
        
        Returns 
        img_info : Image_info : instance of currently selected Image_info
        '''
        return self.image_infos[self.index]
    
    def next_index(self):
        '''
        Move to next index in data frame
        
        Returns
        at_end : bool : return True if further incrementing will get you past the end
        '''
        
        if self.index > self.n_images-1:
            return False
        else: 
            self.index = self.index + 1
            return True
    
    def get_n_reviewed(self):
        #Return the number of images that have been manually reviewed
        n_manual = 0
        for i in self.image_infos: 
            n_manual += i.manually_reviewed
            
        return n_manual
    
    def write(self):
        '''
        Write session file to path
        '''
        df = pd.DataFrame()
        for i in self.image_infos: 
            root_image,file_image = os.path.split(i.img_path)
            root_annotation,file_annotation = os.path.split(i.annotation_path)
            series = {"id":i.id, "root_image":root_image, "file_image":file_image, "root_annotation":root_annotation, "file_annotation":file_annotation, "manually_reviewed":i.manually_reviewed}
            df.append(series,ignore_index=True)
            
        df.to_csv(self.path,index=False)
    
    def __repr__(self):
        string = "Session, path: {p}, df_dim: {d}, current img: {i}".format(p=self.path,d=self.df.shape,i=self.get_process_info())
        return string
        
        
class Category():
    def __init__(self,name,color_hex,color_human,group,button,show_size):
        '''
        Initialize cateogory that will be annotated

        Params
        name        : str : name of cateogory
        color_hex   : str : Hex color of category
        color_human : str : Human name color of category
        group       : int : Group number. Cycling between groups enables more categories than buttons
        button      : str : Button that gives this object when pressed
        show_size   : int : Display size multiplier of this object 
        '''
        self.name = name
        self.color_hex = color_hex
        self.color_human = color_human 
        self.group = group 
        self.button = button 
        self.show_size = show_size
        #print(self)
        
    def __lt__(self,other):
        #make class sortable with .sort()
        return self.group < other.group
        
    def __repr__(self): 
        return "Category(name = {n}, color_hex = {c_he},color_human = {c_hu},group = {g},button = {b},show_size = {s})".format(n=self.name,c_he=self.color_hex,c_hu=self.color_human,g=self.group,b=self.button,s=self.show_size)

class Categories(): 
    def __init__(self,categories,current_group=1):
        '''
        Super objects of categories. Used to organize categories, so you can f.ex. cycling through category groups we can have more categories than buttons 
        
        Params
        categories    : list of Category : All categories
        current_group : int              : The group of categories that is currently selected
        '''
        self.categories = categories
        self.current_group = current_group

        self.highest_group = 1
        for c in categories: 
            if c.group > self.highest_group: 
                self.highest_group = c.group 
        
        self.verify_unique_buttons()

    @classmethod 
    def load_from_file(self,file_path):
        '''
        Make Category and Categories objects from file

        Params
        file_path : str : path to csv file with info in columns: name,color,group_numb,button 
        '''
        df = pd.read_csv(file_path)
        categories = []
        for i in df.index: 
            c = Category(df.loc[i,"name"],df.loc[i,"color_hex"],df.loc[i,"color_human"],int(df.loc[i,"group_numb"]),df.loc[i,"button"],df.loc[i,"show_size"])
            categories.append(c)
        
        categories.sort()

        return Categories(categories)
    
    def verify_unique_buttons(self):
        #Check that no button is used in multiple groups
        button_and_group = []
        for c in self.categories: 
            button_and_group.append(c.button + str(c.group))
        button_and_group = set(button_and_group)
        if len(self.categories) != len(button_and_group):
            raise ValueError("The same button is used in multiple button groups. len(categories) = "+str(len(categories))+" len(button_and_group) = "+str(len(button_and_group)))

    def next_group(self):
        # Move on to next group         
        self.current_group = self.current_group + 1
        if self.current_group > self.highest_group: 
            self.current_group = 1

    def set_group(self,group): 
        '''
        set group number
        
        Params
        group : int : Group number that is set 
        '''
        if not isinstance(group,int): 
            raise ValueError("Supplied group is not integer. Group: "+str(group)+" type: "+str(type(group)))
        if not group > self.highest_group:
            self.current_group = group 

    def get_active_group(self):
        #Get all categories belonging to current group
        out_categories = []
        for c in self.categories: 
            if c.group == self.current_group: 
                out_categories.append(c)
        return out_categories     
   
    def get_category(self,button):
        '''
        Get category str of current group. Return None if not defined
        
        Params
        button : str : button that is pressed
        
        Returns
        name      : str : name of category
        
        '''
        active = self.get_active_group()
        for c in active: 
            if c.button == button: 
                return c.name
        return "None"
        
    def get_categories_names(self):
        '''
        Get a list of all unique categories
        
        Returns
        categories : set of str : All categories
        '''
        categories = []
        for c in self.categories:
            categories.append(c)
        categories = set(categories)
        return categories
    
    def get_show_size(self,category):
        '''
        Get show_size of category. Return 1 if category is not defined
        
        Params
        category : str : str that gives category name
        
        Returns
        show_size : int : Multiplier for object size
        '''
        for c in self.categories: 
            if c.name == category: 
                return c.show_size
        return 1

    def get_info_text_img(self):
        '''
        Return all categories with one line per category for annotating images 
        
        Returns
        out   : list of str    : One information line per object 
        color : list of tupple : rgb color of each line
        '''
        out = []
        color = []
        for c in self.categories: 
            temp = "  "+ c.name 
            out.append(temp)
            color.append(self.hex_to_bgr(c.color_hex))
        return out,color 

    def get_info_text_visual(self):
        '''
        Return all categories with one line per category for annotation program. 
        
        Returns
        out   : str   : One information line per object 
        '''
        out = "\n"
        for c in self.categories:
            temp = "  "+c.name + " = " + c.color_human 
            temp = "{:<25}  ".format(temp)
            out = out + temp +"group "+str(c.group)
            if self.current_group == c.group: 
                out = out + " <-- "+c.button
            out = out + "\n"
        return out 

    def get_color(self,object_name,return_type="human"):
        '''
        Get the color of object corresponding to the name 

        object_name : str : name of object
        return_type : str : one of ["human", "hex", "bgr"]
        '''
        color = None 
        for c in self.categories: 
            if c.name == object_name:
                if return_type == "human": 
                    return c.color_human 
                color = c.color_hex
                break
        
        if color is None: 
            if return_type == "human": 
                return "white"
            color = "#FFFFFF"
        
        if return_type == "hex": 
            return color 
        elif return_type == "bgr": 
            return self.hex_to_bgr(color)
        else: 
            raise ValueError("return_type is set to "+str(return_type)+". That is not one of ['human','hex','bgr']")

    @classmethod 
    def hex_to_bgr(self,value):
        #Converts hex color code into bgr
        value = value.lstrip('#')
        lv = len(value)
        rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        bgr = (rgb[2],rgb[1],rgb[0])
        return bgr 

class Annotation: 
    def __init__(self,file_id,df,manually_reviewed,categories,changelog = "",next_object_id = None): 
        '''
        Class containing annotation data for image

        Params
        file_id           : str          : Name of annotated file 
        df                : pd.Dataframe : Information about annotated objects. Contains colums: ["center_x","center_y","type"] 
        manually_reviewed : bool         : Whether the annotation is manually reviewed or not 
        categories        : Categories   : Information about categories 
        '''
        self.file_id = file_id
        if "_C" in self.file_id: 
            self.id_channel = "C"+self.file_id.rsplit("_C",1)[1]
        else: 
            self.id_channel = ""
        self.df = df 
        
        self.manually_reviewed = manually_reviewed
        if len(self.df)<0: 
            self.manually_reviewed = False

        self.changelog = changelog
        self.next_object_id = next_object_id
       
        #self.df["contour_id"] = self.id_channel + "_" + df["contour_id"].astype('str') 
        self.df["contour_groups"] = ""
        self.contour_groups = None
        self.categories = categories

    def add_contour_groups(self,contour_groups):
        '''
        Add contour groups from channel for more efficient matching of annotations
        '''
        self.contour_groups = contour_groups 
        if self.contour_groups is not None:
            for i in self.df.index: 
                xy = (self.df.loc[i,"center_x"],self.df.loc[i,"center_y"])
                self.df.loc[i,"contour_groups"] = self.contour_groups.get_contour_groups_with_point(xy) 
    
    @classmethod
    def load_from_file(self,path,categories):
        '''
        Create annotation object from file info

        Params
        path       : str        : Path to annotation file
        categories : Categories : Information about categories that are annotated  

        Returns
        annotation : Annotation : Annotation object with file information
        '''
        reviewed_by_human,changelog,df,next_object_id = Annotation.read_annotation_file(path)
        
        file_id = os.path.splitext(os.path.split(path)[1])[0]
        
        annotation = Annotation(file_id,df,reviewed_by_human,categories,changelog,next_object_id)
        
        return annotation
    
    @classmethod
    def read_annotation_file(self,path): 
        '''
        Load annotation file 

        Returns
        reviewed_by_human : bool         : If the file annotation has been reviewed manually, this is true
        changelog         : str          : the file history
        df   	          : pd.DataFrame : Dataframe information about the objects 
        next_objecct_id   : int          : the id of the next object to be added. To make sure that no object in this annotation ever gets the same id-number
        '''
        
        with open(path,'r') as f:
            lines = f.readlines()
        
        for i in range(0,len(lines)):
            if "--changelog--" in lines[i]:
                changelog_line = i
            if "--organoids--" in lines[i]: 
                info_line = i
                break

        if "reviewed_by_human" in lines[0]:
            if "False" in lines[0]:
                reviewed_by_human = False
            elif "True" in lines[0]:
                reviewed_by_human = True
        
        if ("next_org_id" in lines[1]) or ("next_object_id" in lines[1]):
            next_object_id = lines[1].split("=")[1]
            next_object_id = int(next_object_id.strip())

        changelog = lines[changelog_line+1:info_line]
        changelog = "".join(changelog)
        
        annotation_raw = lines[info_line+1:]
        
        if len(annotation_raw) > 1: 
            annotation_raw = "".join(annotation_raw)
            annotation_raw = io.StringIO(annotation_raw)
        
            if "next_org_id" in lines[1]: 
                sep = ","
            else: 
                sep = ";"
            df = pd.read_csv(annotation_raw,sep = sep)
            
            if "org_id" in df.columns: #legacy version of file 
                df["object_id"] = df["org_id"]
                df["center_x"] = df["X"]
                df["center_y"] = df["Y"]
                
            df = df.loc[:,["center_x","center_y","type","object_id"]].copy()
        else:
            df = pd.DataFrame(columns = ["center_x","center_y","type","object_id"])
            
        return reviewed_by_human,changelog,df,next_object_id 

    @classmethod
    def write_annotation_file(self,path,reviewed_by_human,changelog,df,next_object_id):
        '''
        Save annotation file in specific format for this program
        
        Params
        path              : str          : path to place for saving file
        reviewed_by_human : bool         : If the file annotation has been reviewed manually, this is true
        changelog         : str          : the file history
        df   	          : pd.DataFrame : Dataframe information about the objects 
        next_object_id    : int          : the id of the next organoid to be added. To make sure that no organoid in this annotation ever gets the same id-number
        '''
        if next_object_id is None: 
            if df is not None: 
                if len(df.index) > 0: 
                    next_object_id = pd.to_numeric(df['object_id']).max() + 1
            else: 
                next_object_id = 0
        
        try: 
            next_object_id = int(next_object_id)
        except ValueError: 
            pass

        with open(path,'w') as f:
            if reviewed_by_human: 
                f.write("reviewed_by_human = True\n")
            else:
                f.write("reviewed_by_human = False\n")
            f.write("next_object_id = "+str(next_object_id)+"\n")
            f.write("--changelog--\n")
            f.write(changelog)
            f.write("--organoids--\n")
            
        if df is not None: 
            if len(df.index)>0: 
                df.to_csv(path,mode="a",sep=";",index=False)

        if VERBOSE: print("\tSaved "+str(len(df.index))+" annotations to: "+path)

    def get_points(self,groups=None):
        '''
        Get all annotations 

        Params
        groups : list of str : If this value is passed, only returns points that belongs to the contour group with this name  
        
        Returns
        df     : pd.dataframe : Dataframe with point values
        '''
        if groups is None or groups == []: 
            return self.df
        else:
            out = [False]*len(self.df.index)
            for group in groups:
                temp = list(self.df['contour_groups'].str.contains(group))
                out = [a or b for a, b in zip(out, temp)]
            return self.df.loc[out,]
    
    def __repr__(self):
        return "{class_str}: file_id {f}, len(df) = {df_n}".format(class_str=self.__class__.__name__,f=self.file_id,df_n = len(self.df.index))
