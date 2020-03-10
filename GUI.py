import tkinter as tk
import tkinter.filedialog as fdialog
import tkinter.messagebox as mbox 
import os

class mainWindow:

    def __init__(self,master):
        
        self.this_script_path = os.path.abspath(os.path.split(__file__)[0])
        self.boco_visual_path = os.path.join(self.this_script_path,"boco_visual.py")
        self.coco_visual_path = os.path.join(self.this_script_path,"coco_visual.py")

        self.session_file = None
        self.scale_image = None
        self.from_first = None

        self.master = master
        
        self.boco_or_coco_label = tk.Label(master,text="\nRun boco or coco:")
        self.boco_or_coco_label.pack()

        self.boco_or_coco_text = tk.StringVar(master)
        self.boco_or_coco_text.set("boco")
        self.boco_or_coco_entry = tk.OptionMenu(master,self.boco_or_coco_text,"coco","boco")
        self.boco_or_coco_entry.pack()
        
        self.file_path_label=tk.Label(master,text="File path to session file: ")
        self.file_path_label.pack()

        self.file_path_entry_text = tk.StringVar()
        self.file_path_entry=tk.Entry(master,textvariable=self.file_path_entry_text,width=150)
        self.file_path_entry.pack()
        
        self.find_file=tk.Button(master,text='Find file',command=self.load_file)
        self.find_file.pack()
        
        self.scale_image_label = tk.Label(master,text="\nImage height:") 
        self.scale_image_label.pack()

        self.scale_image_entry_text = tk.StringVar()
        self.scale_image_entry_text.set("1080")
        self.scale_image_entry = tk.Entry(master,textvariable=self.scale_image_entry_text,width=5)
        self.scale_image_entry.pack()
        
        self.from_first_label = tk.Label(master,text="\nStart from image:")
        self.from_first_label.pack()

        self.from_first_text = tk.StringVar(master)
        self.from_first_text.set("Last edited")
        self.from_first_entry = tk.OptionMenu(master,self.from_first_text,"First image","Last edited")
        self.from_first_entry.pack()
        
        self.run_label = tk.Label(master,text="\n")
        self.run_label.pack()

        self.run_button = tk.Button(master,text="RUN!",command=self.run)
        self.run_button.pack()
    
    def run(self): 
        if self.get_values(): 
            self.run_visual()
    
    def run_visual(self):
        if self.use_boco: 
            cmd = "python \"{o}\" --session_file \"{f}\" --height {s}".format(o=self.boco_visual_path,f=self.session_file,s=self.scale_image)
        else: 
            cmd = "python \"{o}\" --session_file \"{f}\" --height {s}".format(o=self.coco_visual_path,f=self.session_file,s=self.scale_image)
        if self.from_first: 
            cmd = cmd + " --from_first"
        print(cmd)
        exit_val = os.system(cmd)
        if exit_val != 0: 
            mbox.showerror("The visual editor had an error","Command did not finish properly! Check command output.")
        
    def cleanup(self):
        self.value=self.e.get()
        self.master.destroy()

    def load_file(self):
        file_path = fdialog.askopenfilename(filetypes=(("Text files", "*.txt"),
                                           ("All files", "*.*") ))
        self.file_path_entry_text.set(str(file_path))
    
    def get_values(self): 
        scale_image = self.scale_image_entry.get()
        try: 
            self.scale_image = int(scale_image)
        except: 
            mbox.showerror("scale_image error","could not convert scale_image to int. scale_image set to: "+str(scale_image))
            return False
        
        boco_or_coco = self.boco_or_coco_text.get()
        if boco_or_coco == "boco": 
            self.use_boco = True
        elif boco_or_coco == "coco": 
            self.use_boco = False 
        else: 
            mbox.showerror("boco or coco error","Not a valid selection of boco_or_coco. Currently set to: "+str(boco_or_coco))
            return False 
        
        from_first = self.from_first_text.get()
        if from_first == "First image": 
            self.from_first = True
        elif from_first =="Last edited": 
            self.from_first = False
        else: 
            mbox.showerror("from_first error","from_first need to be a logical value. Currently set to: "+str(from_first))
            return False 
    
        session_file = self.file_path_entry.get()
        if not os.path.exists(session_file):
            mbox.showerror("session_file error","Can't find specified file: "+str(session_file))
            return False 
        else: 
            self.session_file = session_file 
        
        return True

root=tk.Tk()
m=mainWindow(root)
root.mainloop()
