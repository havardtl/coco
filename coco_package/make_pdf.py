# This file contains classes to make pdfs that display info from images

import cv2
import os
import math
import pandas as pd

VERBOSE = False

def set_verbose(): 
    VERBOSE = True

class Image_in_pdf:
    
    def __init__(self,x,y,img_path,data,x_vars,y_vars,image_vars,processed_folder,goal_img_dim,max_size=5000*5000):
        '''
        Information about single image that is needed to make Pdf document
        
        Params
        x                : int          : Relative x position of image. zero-indexed
        y                : int          : Relative y position of image. zero-indexed
        data             : dict         : Dictionary with all the metadata about image to plot in annotation boxes around images
        x_vars           : x_vars       : Variables in "data" to plot on x-axis annotation fields 
        y_vars           : y_vars       : Variables in "data" to plot on y-axis annotation fields 
        Image_vars       : list of str  : Variables in "data" to plot on top of image
        processed_folder : str          : Folder with images that have right dimensions and ratio
        goal_img_dim     : tuple of int : (y,x) dimensions that specifies the ratio image must be. 
        max_size         : int          : maximal size of image in MP. If it is bigger than this it is reduced to that size. 
        '''
        self.x = x
        self.y = y
        self.img_path = img_path
        self.data = data
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.image_vars = image_vars
        
        self.file_id = os.path.splitext(os.path.split(self.img_path)[1])[0]
        self.processed_path = os.path.join(processed_folder,self.file_id+".jpg")
        
        self.goal_img_dim = goal_img_dim
        self.goal_ratio = float(self.goal_img_dim[1])/float(self.goal_img_dim[0])
        
        self.max_size = max_size 
        
        self.img = None 
        self.img_dim = None
        self.old_dim = None
        
        self.changed_image = False 
        
        if not os.path.exists(self.processed_path):
            self.img = cv2.imread(self.img_path)
            self.img_dim = self.img.shape
            self.old_dim = self.img_dim
            self.to_max_size()
            self.to_right_ratio()
            self.write_image()
        
        #if VERBOSE: print(self)
        
    def write_image(self): 
        #Write image to processed_path, change image path and delete from memory
        if self.changed_image: 
            if not cv2.imwrite(self.processed_path,self.img):
                raise ValueError("Could not save image to: "+self.processed_path)
            self.img_path = self.processed_path
        self.img = None
    
    def to_right_ratio(self):
        #If image has a different ratio than Crop image so that it has the goal_img_dim ratio. 
        img_ratio = float(self.img_dim[1])/float(self.img_dim[0])
        if not ((self.goal_ratio - 0.01) < img_ratio < (self.goal_ratio + 0.01)):
            #if VERBOSE: print("Changing ratio")
            self.changed_image = True
            
            if img_ratio < self.goal_ratio:
                new_width = int(self.goal_ratio*float(self.img_dim[0]))
                new_height = int(self.img_dim[0])
            else:
                new_width  = int(self.img_dim[1])
                new_height = int((1/self.goal_ratio)*float(self.img_dim[1]))
                
            right_ratio_rectangle = Rectangle([0,0,new_width,new_height])
            self.img = Channel.imcrop(self.img,right_ratio_rectangle,value=(150,150,150))
            self.img_dim = self.img.shape
    
    def to_max_size(self):
        #If image is to big, convert it to self.max_size
        img_size = self.img_dim[0]*self.img_dim[1]
        
        if img_size > self.max_size: 
            #if VERBOSE: print("Shrinking to max size")
            self.changed_image = True
            ratio = np.sqrt(self.max_size)/np.sqrt(img_size)
            new_dim = (int(self.img_dim[1]*ratio),  int(self.img_dim[0]*ratio))
            
            self.img = cv2.resize(self.img,new_dim)
            self.img_dim = self.img.shape

    def __repr__(self):
        string = "{class_str}: path = {p}, xy = ({x},{y}), old_dim = {od}, new_dim = {nd},goal_img_dim = {gd}".format(class_str = self.__class__.__name__,p=self.img_path,x = self.x,y = self.y, od = self.old_dim,nd = self.img_dim,gd = self.goal_img_dim)
        return string

class Pdf:
    
    def __init__(self,save_path,images_in_pdf):
        '''
        Plot images in a pdf sorted by info about images
        
        Params
        save_path     : str                  : Path to save pdf in
        images_in_pdf : list of Image_in_pdf : List of objects with information about image path, location in pdf and x and y variables to plot
        '''
        
        self.save_path = save_path
        self.images_in_pdf = images_in_pdf
        
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
        
        avg_character_width = 3
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
        canvas.setFont("Helvetica",5)
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
        
        if VERBOSE: print("Saving PDF to: "+self.save_path)

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
        goal_aspect_ratio = self.images_in_pdf[0].goal_img_dim[0]/self.images_in_pdf[0].goal_img_dim[1]

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
    
    @classmethod
    def img_dim_str_to_tuple(self,img_dim_str): 
        #image dimensions stored as string version of tupple in dataframe, convert it to tupple int here
        img_dim_str = img_dim_str.replace("(","").replace(")","")
        img_dim = img_dim_str.split(", ",1)
        img_dim = (int(img_dim[0]),int(img_dim[1]))
        return img_dim

    @classmethod
    def plot_images_pdf(self,save_folder,df,file_vars,x_vars,y_vars,image_vars,image_dim,processed_folder,sort_ascending=None,max_size=5000*5000):
        '''
        Create PDFs with images grouped after variables
        
        Params
        save_folder      : str          : Folder where pdf is saved
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
            self.make_pdf_from_df(save_path,df_f,x_vars,y_vars,image_vars,processed_folder,image_dim,max_size)
    
    @classmethod
    def make_pdf_from_df(self,save_path,df,x_vars,y_vars,image_vars,processed_folder,image_dim,max_size=5000*5000): 
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
            pdf_imgs.append(Image_in_pdf(x,y,full_path,data,x_vars,y_vars,image_vars,processed_folder,image_dim,max_size))

        pdf = Pdf(save_path,pdf_imgs)
        pdf.make_pdf()

