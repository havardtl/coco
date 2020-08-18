#!/usr/bin/env python3
import argparse

#######################
# Argument parsing
#######################
parser = argparse.ArgumentParser(description = 'Run visual classification of images. Manually determine classes based on right and left click with mouse.')
parser.add_argument('--session_file',default='ORGAI_session_file.txt',help='session file. Create a new one with ORGAI_init.py')
parser.add_argument('--height',type=int,default=1024,help='Resize image to this height. Keep aspect ratio')
parser.add_argument('--zoom',type=float,default=1,help='number of pixels to show per window pixel when using zoom mode')
parser.add_argument('--epsilon',type=float,default=5,help='Size of rectangles for each object')
parser.add_argument('--from_first',action='store_true',help='Default is to start from the last manual reviewed, with this switch you start from first')
parser.add_argument('--categories',type=str,help='File to load category information from. Default is to load it from default file in config/boco_categories.csv')
args = parser.parse_args()

##############################
# Dependencies and variables
##############################
import os 
from coco_package import visual_editor
from coco_package import info

############################
# Run program
############################

this_script_folder = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if args.categories is None: 
    args.categories = os.path.join(this_script_folder,"config","boco_categories.csv")

categories = info.Categories.load_from_file(args.categories)
current_img = visual_editor.Current_Img(args.session_file,args.from_first,args.height,args.epsilon,args.zoom)

mainwindow = visual_editor.MainWindow(categories,current_img)

