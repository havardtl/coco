import tkinter as tk
from PIL import Image, ImageTk
import os 
import pandas as pd
import datetime
import math
import copy

import classes.classes as classes

class Zoomed_Img():

    def __init__(self,img_path,window_height,zoom_level): 
        '''
        Class for handling rescaling and zooming of image

        Params
        img_path      : str : Path to original image
        window_height : int : output height of window. Width is scaled to image
        zoom_level    : float   : Number of pixels to show per pixel when using zoom mode 
        '''
        self.zoom_mode = False 
        self.segments_str = "  Overview mode"
        self.zoom_level = zoom_level
        
        self.img_path = img_path 
        self.original_img = Image.open(self.img_path)
        self.width_original,self.height_original = copy.copy(self.original_img.size)
        
        self.width_zoom = self.width_original * self.zoom_level
        self.height_zoom = self.height_original * self.zoom_level 
        
        self.zoomed_img = self.original_img.resize((int(self.width_zoom),int(self.height_zoom))) 
        
        self.height_window = window_height
        self.width_window = (self.width_original/self.height_original) * self.height_window
        
    def switch_zoom_mode(self):
        #Change between zoom mode and non-zoom mode
        self.zoom_mode = not self.zoom_mode
        if self.zoom_mode: 
            self.make_segments()
        
        self.update_segments_str()
    
    def zoom_move_right(self): 
        #Move one zoom segment to the right
        if self.current_zoom_segment[0]+1 < self.x_n: 
            self.current_zoom_segment[0] = self.current_zoom_segment[0] + 1
            self.update_segments_str()

    def zoom_move_left(self): 
        #Move one zoom segment to the left
        if self.current_zoom_segment[0]-1 >= 0: 
            self.current_zoom_segment[0] = self.current_zoom_segment[0] - 1
            self.update_segments_str()

    def zoom_move_up(self): 
        #Move one zoom segment up
        if self.current_zoom_segment[1]-1 >= 0: 
            self.current_zoom_segment[1] = self.current_zoom_segment[1] - 1
            self.update_segments_str()

    def zoom_move_down(self):
        #Move one zoom segment down
        if self.current_zoom_segment[1]+1 < self.y_n: 
            self.current_zoom_segment[1] = self.current_zoom_segment[1] + 1
            self.update_segments_str()
            
    def zoom_in_point(self,x,y): 
        '''
        Zoom in on segment that contains window x,y values. Only do so if in overview mode. 
        
        Params
        x : int : window x value
        y : int : window y value 
        '''
        self.switch_zoom_mode()
        if self.zoom_mode: 
            x = x * (self.width_zoom/self.width_window)
            y = y * (self.height_zoom/self.height_window)
            
            def inner_function(): 
                #wrapper function to enable return out of nested loop 
                for xi in range(self.x_n): 
                    x0 = (xi+1) * self.x_step
                    if x0 > x: 
                        for yi in range(self.y_n): 
                            y0 = (yi+1) * self.y_step
                            if y0 > y:
                                self.current_zoom_segment = [xi,yi]
                                return 
            inner_function()
            self.update_segments_str()

    def make_segments(self):
        #Divide image into segments based on zoom_level
        x,y = self.pixel_to_window(self.width_original,self.height_original,zoom_mode = False)
        
        x_ratio = self.width_zoom/x
        y_ratio = self.height_zoom/y

        self.x_n = math.ceil(x_ratio)
        self.y_n = math.ceil(y_ratio)
        
        self.x_step = self.width_zoom/self.x_n 
        self.y_step = self.height_zoom/self.y_n 
        
        self.current_zoom_segment = [0,0]
        
    def update_segments_str(self):
        #Update string that tells you were you are zoomed in in the image
        if not self.zoom_mode: 
            self.segments_str =  "  Overview mode"
            for i in range(self.y_n): 
                self.segments_str = self.segments_str + "\n"
        else:
            self.segments_str = "  "
            for i in range(self.y_n): 
                for j in range(self.x_n): 
                    if i == self.current_zoom_segment[1] and j == self.current_zoom_segment[0]: 
                        self.segments_str = self.segments_str + "X "
                    else: 
                        self.segments_str = self.segments_str + "O "
                self.segments_str = self.segments_str + "\n  "
            
    def get_display_img(self):
        #Get part of image that shall be displayed and in correct resolution
        if not self.zoom_mode: 
            return self.original_img.resize((int(self.width_window),int(self.height_window)))
        else:
            self.x0 = self.current_zoom_segment[0]*self.x_step 
            self.y0 = self.current_zoom_segment[1]*self.y_step
            self.x1 = self.x0 + self.x_step
            self.y1 = self.y0 + self.y_step
            display_img = self.zoomed_img.crop((self.x0,self.y0,self.x1,self.y1))
            display_img.load()
            return display_img
  
    def window_to_pixel(self,x,y,zoom_mode = None): 
        '''
        Translate window coordinates to pixels
        
        Params
        x         : int  : x in window reference
        y         : int  : y in window reference
        zoom_mode : bool : Specify zoom mode instead of using currently set one 

        Returns
        x : int : x in pixel reference
        y : int : y in pixel reference
        '''
        if zoom_mode is None: 
            zoom_mode = self.zoom_mode
        
        if not zoom_mode: 
            x = x * (self.width_original/self.width_window)
            y = y * (self.height_original/self.height_window)
        else: 
            x = self.current_zoom_segment[0] * (self.x_step/self.zoom_level) + (x / self.zoom_level) 
            y = self.current_zoom_segment[1] * (self.y_step/self.zoom_level) + (y / self.zoom_level)
        
        return x,y
        
    def pixel_to_window(self,x,y,zoom_mode = None):
        '''
        Translate pixel coordinates to pixels
        
        Params
        x         : int  : x in pixel reference
        y         : int  : y in pixel reference
        zoom_mode : bool : Specify zoom mode instead of using currently set one 
        
        Returns
        x : int : x in window reference
        y : int : y in window reference
        '''
        if zoom_mode is None: 
            zoom_mode = self.zoom_mode
        
        if not zoom_mode: 
            x = x * (self.width_window/self.width_original)
            y = y * (self.height_window/self.height_original)
        else:
            x = (x * self.zoom_level) - self.current_zoom_segment[0] * self.x_step
            y = (y * self.zoom_level) - self.current_zoom_segment[1] * self.y_step
        return x,y

    def epsilon_in_window(self,epsilon): 
        '''
        Get epsilon in correct zoom level 

        Params
        epsilon : int : input epsilon in window mode

        Returns
        epsilon : int : epsilon in pixels 
        '''
        if not self.zoom_mode: 
            return epsilon / (self.width_original/self.width_window)
        else: 
            return epsilon * self.zoom_level

class Current_Img():

    def __init__(self,session_path,from_first,window_height,epsilon,zoom_level): 
        '''
        class containing info about displayed image and its annotation 
        
        Params
        session_path  : str   : Session class this image belongs to
        from_first    : bool  : If True, start from first image rather than starting from last manually reviewed
        window_height : int   : Height of window display
        epsilon       : int   : area around annotated object that cannot have other objects in image pixels
        zoom_level    : float : Number of pixels to show per pixel when using zoom mode 
        '''
        self.session = Session(session_path,from_first)
        self.window_height = window_height
        self.epsilon = epsilon
        self.zoom_level = zoom_level
        
        now = datetime.datetime.now()
        self.today = now.strftime("%Y-%m-%d-%H:%M")
        
        self.annot_path = None
        self.img_path,self.annot_path = self.session.get_img_and_annot_paths()
        self.load_annotation()
        self.zoomed_img = Zoomed_Img(self.img_path,self.window_height,self.zoom_level)
        print(self.session.get_counter()+"\t"+self.annot_path + "\t is displayed",flush=True)
    
    def update(self,change):
        '''
        Update state to fit current variables

        change : int : increment to update by. -1 = back, 0 = the same, 1 = forward
        '''
        self.save_annotation()
        
        changed_index = self.session.update(change)
        
        if changed_index: 
            self.img_path,self.annot_path = self.session.get_img_and_annot_paths()
            if not os.path.exists(self.img_path): 
                raise ValueError("Image path does not exist: "+self.img_path)
            if not os.path.exists(self.annot_path): 
                raise ValueError("Annotation path does not exist: "+self.annot_path)
            
            self.load_annotation()
            
            self.zoomed_img = Zoomed_Img(self.img_path,self.window_height,self.zoom_level)
            
        print(self.session.get_counter()+"\t"+self.annot_path + "\t is displayed",flush=True)
    
    def save_all(self): 
        # Save annotations and session file
        self.save_annotation()
        self.session.save_session_file(verbose=True)
    
    def switch_zoom_mode(self):
        self.zoomed_img.switch_zoom_mode()
    
    def zoom_move_right(self): 
        self.zoomed_img.zoom_move_right()
       
    def zoom_move_left(self): 
        self.zoomed_img.zoom_move_left()
    
    def zoom_move_up(self): 
        self.zoomed_img.zoom_move_up()
    
    def zoom_move_down(self): 
        self.zoomed_img.zoom_move_down()
        
    def zoom_in_point(self,x,y): 
        #zoom into a point
        self.zoomed_img.zoom_in_point(x,y)
    
    def get_zoom_mode(self): 
        return self.zoomed_img.zoom_mode

    def get_zoom_info(self):
        return self.zoomed_img.segments_str

    def get_img(self):
        #Return rescaled PIL image
        return self.zoomed_img.get_display_img()
    
    def pixel_to_window(self,x,y):
        return self.zoomed_img.pixel_to_window(x,y)

    def window_to_pixel(self,x,y):
        return self.zoomed_img.window_to_pixel(x,y)

    def get_rectangles(self):
        '''
        Translate annotation into coordinates of rectangles

        Returns 
        rectangles     : list of tuples : Outer corners of rectangles in a tupple, i.e. [(x1,y1,x2,y2),(x1,y1,x2,y2),...]
        classification : list of str    : classification of object rectangles represent
        '''
        epsilon = self.zoomed_img.epsilon_in_window(self.epsilon)
        
        rectangles = []
        classification = []
        for i in self.annotation.index:
            x,y = self.pixel_to_window(self.annotation.loc[i,"center_x"],self.annotation.loc[i,"center_y"])
            rectangles.append((x-epsilon,y-epsilon,x+epsilon,y+epsilon))
            classification.append(self.annotation.loc[i,"type"])
            
        return rectangles,classification

    def save_annotation(self):
        #Save annotation file
        classes.Annotation.write_annotation_file(self.annot_path,self.reviewed_by_human,self.changelog,self.annotation,self.next_object_id)
        print("\tSaved annotation to "+self.annot_path,flush=True)
        
    def load_annotation(self):
        #Load annotation file 
        self.reviewed_by_human,self.changelog,self.annotation,self.next_object_id = classes.Annotation.read_annotation_file(self.annot_path)
        self.reviewed_by_human = True
        self.changelog = self.changelog + self.today + " manually_reviewed\n"
        self.id = os.path.split(self.annot_path)[1]
        self.id = os.path.splitext(self.id)[0]
        
    def euc_dist(self,x1,y1,x2,y2):
        #calculate euclidian distance
        return(math.sqrt((x2-x1)**2 + (y2-y1)**2))    
        
    def add_object(self,x,y,classification,erase,show_size,erase_multiplier=3):
        '''
        Remove objects that are closer than epsilon distance and add object to annotation 

        Params
        x                : int  : x position in pixel reference
        y                : int  : y position in pixel reference
        classification   : str  : classification of object
        erase            : bool : Do not add object, only erase nearby ones
        show_size        : int  : multiplier for epsilon to determine editing area
        erase_multiplier : int  : Factor to multiply with if in erase mode 
        '''
        epsilon = self.epsilon * self.zoom_level * show_size
        if erase:
            epsilon = epsilon * erase_multiplier 
        distant = []
        for i in self.annotation.index:
            d = self.euc_dist(x,y,self.annotation.loc[i,"center_x"],self.annotation.loc[i,"center_y"])
            if d > (epsilon):
                distant.append(True)
            else: 
                distant.append(False)
        self.annotation = self.annotation[distant]
        
        if not erase: 
            data = {'id':[self.id],'center_x':[x],'center_y':[y],'type':[classification],'source':['manual'],'object_id':self.next_object_id}
            self.next_object_id = self.next_object_id + 1
            temp = pd.DataFrame(data=data)
            self.annotation = self.annotation.append(temp,ignore_index=True,sort=False)
        
class Session(): 
    
    def __init__(self,session_file,from_first): 
        '''
        Class containing info about all images and which one is selected

        Params
        session_file : str  : path to session file
        from_first   : bool : Start from first image instead of starting from the last reviewed
        '''
        self.session_file = session_file
        self.main_folder = os.path.split(self.session_file)[0]
        
        print("Loading session file from: "+self.session_file,flush=True)
        self.df = pd.read_csv(self.session_file)
        print("Found "+str(len(self.df.index))+" images. Of wich "+str(self.df["manually_reviewed"].sum())+" is already reviewed",flush=True)
        
        if from_first: 
            self.index = self.df.index[0]
        else: 
            self.index = self.df.index[-1]
            for i in self.df.index:
                if not self.df.loc[i,'manually_reviewed']:
                    self.index = i
                    break
        
    def update(self,change):
        '''
        Update state to fit current variables
        
        Params
        change : int : increment to update by. -1 = back, 0 = the same, 1 = forward
        
        Returns
        changed_index : bool : If True, index was changed
        '''
        self.df.loc[self.index,"manually_reviewed"] = True
        self.save_session_file() 
        
        changed_index = False 
        next_index = self.index + change
        if not (next_index > self.df.index.max() or next_index < 0):
            self.index = next_index
            changed_index = True
        
        return changed_index
    
    def get_img_and_annot_paths(self): 
        '''
        Get path to image and annotation
        
        Returns 
        img_path   : str : Path to current image
        annot_path : str : Path to current annotation
        '''
        img_path = os.path.join(self.main_folder,self.df.loc[self.index,"root_image"],self.df.loc[self.index,"file_image"])
        annot_path = annot_path = os.path.join(self.main_folder,self.df.loc[self.index,"root_annotation"],self.df.loc[self.index,"file_annotation"])
        
        return img_path,annot_path
    
    def save_session_file(self,verbose=False):
        #Save session file to session file path
        self.df.to_csv(self.session_file,index=False)
        if verbose: print("\tSaved session file to "+str(self.session_file),flush=True)
        
    def get_counter(self):
        # Return string describing which image we are on
        return str(self.index+1)+"/"+str(len(self.df.index))
    
class MainWindow():

    def __init__(self,categories,current_img):
        '''
        Initialize main window of program with image that shall be annotated

        categories  : Categories     : Object with information about categories that is being annotated
        '''
        self.main = tk.Tk()
        
        self.categories = categories
        self.current_img = current_img
        self.erase_mode = False
        
        pil_img = self.current_img.get_img()
        width,height = pil_img.size
        
        self.canvas = tk.Canvas(self.main,width=width,height=height)
        self.img = ImageTk.PhotoImage(pil_img)
        self.img_on_canvas = self.canvas.create_image(0,0,image=self.img,anchor='nw')
        
        self.main.bind('<Left>',self.key_left)
        self.main.bind('<Right>',self.key_right)
        self.main.bind('<KeyPress-space>',self.key_set_erase_mode)
        self.main.bind('<KeyRelease-space>',self.key_unset_erase_mode)
        self.main.bind('r',self.key_save)
        self.main.bind('<Button-1>',self.key_mouse_left)
        self.main.bind('<Button-3>',self.key_mouse_right)
        self.main.bind('<Button-2>',self.key_mouse_middle)
        self.main.bind('q',self.key_switch_group)
        self.main.bind('z',self.key_zoom_mode)
        self.main.bind('w',self.key_zoom_move_up)
        self.main.bind('a',self.key_zoom_move_left)
        self.main.bind('s',self.key_zoom_move_down)
        self.main.bind('d',self.key_zoom_move_right)
        self.main.bind('1',self.key_set_group_1)
        self.main.bind('2',self.key_set_group_2)
        self.main.bind('3',self.key_set_group_3)
        self.main.bind('4',self.key_set_group_4)
        self.main.bind('5',self.key_set_group_5)
        self.main.bind('6',self.key_set_group_6)
        self.main.bind('7',self.key_set_group_7)
        self.main.bind('8',self.key_set_group_8)
        self.main.bind('9',self.key_set_group_9)

        self.main.title('Manual object annotater')
        
        self.rectangles = []
        
        self.info_window = tk.Toplevel(self.main)
        self.info_window.wm_title("Info window")
        self.button_info_text = "  q = switch groups\n  hold down spacebar = erase mode\n  z = zoom_mode\n  r = save\n  wasd = move in zoom mode\n  mouse_middle = zoom in on mouse point\n  1,2,3... = jump to group with number"
        self.info = tk.Label(self.info_window,anchor = "w",text="",justify="left",bg="black",fg ="white" )
        self.info.pack(fill="both",anchor = "w",expand=True,padx=0,pady=0)
        
        self.erase_mode_info = self.canvas.create_text(0,0,anchor="nw",fill="red",text="")
        
        self.canvas.pack()
        self.draw()

        self.main.mainloop()

    def draw(self):
        '''
        Draw objects in window 
        '''
        pil_img = self.current_img.get_img() 
        self.img = ImageTk.PhotoImage(pil_img)
        self.canvas.itemconfig(self.img_on_canvas,image=self.img)
        
        width,height = pil_img.size 
        self.canvas.config(width=width,height=height)

        for item in self.rectangles:
            self.canvas.delete(item)
        
        rectangles,classifications = self.current_img.get_rectangles()
        
        self.rectangles = []
        for i in range(0,len(rectangles)):
            r = rectangles[i]
            color = self.categories.get_color(classifications[i])
            show_size = self.categories.get_show_size(classifications[i])
            self.rectangles.append(self.canvas.create_rectangle(r[0]-show_size,r[1]-show_size,r[2]+show_size,r[3]+show_size,outline=color,width=show_size))
        
        info_text = self.categories.get_info_text_visual() + "\n\n"+self.button_info_text +"\n\n" + self.current_img.get_zoom_info()
        self.info.config(text = info_text)

        if self.erase_mode:
            self.canvas.itemconfig(self.erase_mode_info,text=" ERASE MODE")
        else:
            self.canvas.itemconfig(self.erase_mode_info,text="")

    def update(self,change):
        '''
        Update state to fit current variables

        change : int : increment to update by. -1 = back, 0 = the same, 1 = forward
        '''
        self.current_img.update(change)
        
    def key_left(self,event):
        #Move one image to the left
        self.update(change=-1)
        self.draw()

    def key_right(self,event):
        #Move one image to the right
        self.update(change=1)
        self.draw()

    def key_save(self,event):
        #Save state in files
        self.current_img.save_all()

    def key_mouse_left(self,event):
        #Add mouse_left category to position
        x,y = self.current_img.window_to_pixel(event.x,event.y)
        name = self.categories.get_category("mouse_left")
        show_size = self.categories.get_show_size(name)
        self.current_img.add_object(x,y,name,self.erase_mode,show_size)
        self.draw()

    def key_mouse_right(self,event):
        #Add mouse_right category to position
        x,y = self.current_img.window_to_pixel(event.x,event.y)
        name = self.categories.get_category("mouse_right")
        show_size = self.categories.get_show_size(name)
        self.current_img.add_object(x,y,name,self.erase_mode,show_size)
        self.draw()

    def key_mouse_middle(self,event):
        #Zoom in on point in image
        self.current_img.zoom_in_point(event.x,event.y)
        self.draw()

    def key_switch_group(self,event):
        #Change category group
        self.categories.next_group()
        self.draw()
     
    def key_set_group_1(self,event): 
        #Set group to group 1
        self.categories.set_group(1)
        self.draw()

    def key_set_group_2(self,event): 
        #Set group to group 2
        self.categories.set_group(2)
        self.draw()

    def key_set_group_3(self,event): 
        #Set group to group 3
        self.categories.set_group(3)
        self.draw()

    def key_set_group_4(self,event): 
        #Set group to group 4
        self.categories.set_group(4)
        self.draw()

    def key_set_group_5(self,event): 
        #Set group to group 5
        self.categories.set_group(5)
        self.draw()

    def key_set_group_6(self,event): 
        #Set group to group 6
        self.categories.set_group(6)
        self.draw()

    def key_set_group_7(self,event): 
        #Set group to group 7
        self.categories.set_group(7)
        self.draw()

    def key_set_group_8(self,event): 
        #Set group to group 8
        self.categories.set_group(8)
        self.draw()

    def key_set_group_9(self,event): 
        #Set group to group 9
        self.categories.set_group(9)
        self.draw()

    def key_set_erase_mode(self,event):
        #Switch into erase mode
        self.erase_mode = True 
        self.canvas.itemconfig(self.erase_mode_info,text=" ERASE MODE")
        #self.erase_mode = not self.erase_mode

    def key_unset_erase_mode(self,event):
        #Switch out of erase mode
        self.erase_mode = False
        self.canvas.itemconfig(self.erase_mode_info,text="")
        
    def key_zoom_mode(self,event):
        #Switch in and out of zoom mode
        self.current_img.switch_zoom_mode()
        self.draw()

    def key_zoom_move_right(self,event):
        if self.current_img.get_zoom_mode():
            self.current_img.zoom_move_right()
            self.draw()

    def key_zoom_move_left(self,event):
        if self.current_img.get_zoom_mode():
            self.current_img.zoom_move_left()
            self.draw()

    def key_zoom_move_up(self,event):
        if self.current_img.get_zoom_mode():
            self.current_img.zoom_move_up()
            self.draw()

    def key_zoom_move_down(self,event):
        if self.current_img.get_zoom_mode():
            self.current_img.zoom_move_down()
            self.draw()
