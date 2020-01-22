#!/usr/bin/env python3
import argparse

#######################
# Argument parsing
#######################
parser = argparse.ArgumentParser(description = 'Run visual classification of images. Manually determine classes based on right and left click with mouse.')
parser.add_argument('--session_file',default='coco_session_file.txt',help='session file linking annotations and images.')
parser.add_argument('--projections',default='graphical/projections_raw',help='Raw projections to fall back to in case session file is not available')
parser.add_argument('--annotations',default='annotations',help='Where to put annotation files if no session file is not available')
parser.add_argument('--height',type=int,default=1300,help='Resize image to this height. Keep aspect ratio')
parser.add_argument('--zoom',type=float,default=2,help='number of pixels to show per window pixel when using zoom mode')
parser.add_argument('--epsilon',type=float,default=2,help='Size of rectangles for each object')
parser.add_argument('--from_first',action='store_true',help='Default is to start from the last manual reviewed, with this switch you start from first')
parser.add_argument('--categories',type=str,help='File to load category information from. Default is to load it from default file in utilities/categories.csv')
parser.add_argument('--filter',type=str,default=None,help='Only include files that have these strings when making session file. Separate multiple filters with comma. With multiple filters all files with either or both filters are included.')
args = parser.parse_args()

#######################
# Dependencies
#######################
import os
import pandas as pd

#######################
# Run program
#######################
this_script_folder = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if args.categories is None: 
    args.categories = os.path.join(this_script_folder,"utilities","categories.csv")
    
if not os.path.exists(args.session_file): 
    print("Session file does not exist. Creating one in: "+args.session_file)
    os.makedirs(args.annotations,exist_ok=True)
    img_files = os.listdir(args.projections)
    
    if args.filter is not None: 
        filters = args.filter.split(",")
        
        imgs_keep = []
        for i in range(len(img_files)): 
            keep = False 
            for f in filters: 
                if f in img_files[i]: 
                    keep = True
            if keep: 
                imgs_keep.append(img_files[i])
        
        print("Found {n_pre} images. Only keeping those that have string: {f}, giving {n_post} images.".format(n_pre=len(img_files),f=filters,n_post=len(imgs_keep)))
        img_files = imgs_keep
        
    if len(img_files) < 1: 
        raise ValueError("Cannot make a session file from no image files. len(img_files) = "+str(len(img_files)))
    
    ids = []
    text_files = []
    for f in img_files: 
        file_id = os.path.splitext(f)[0]
        ids.append(file_id)
        text_files.append(file_id+".txt")
    df = pd.DataFrame({"id":ids,"root_image":args.projections,"file_image":img_files,"root_annotation":args.annotations,"file_annotation":text_files,"manually_reviewed":False})

    for i in df.index:
        annot_path = os.path.join(df.loc[i,"root_annotation"],df.loc[i,"file_annotation"])
        if not os.path.exists(annot_path): 
            with open(annot_path,"w") as f: 
                f.write("reviewed_by_human = False\nnext_org_id = 0\n--changelog--\n--organoids--\nid,X,Y,type,source,org_id,area,centroid_x,centroid_y,perimeter,hull_tot_area,hull_defect_area,solidity,radius_enclosing_circle,equivalent_diameter,circularity,mean_grey,integrated_density\n")
    
    df.to_csv(args.session_file,index=False)

cmd = "ORGAI_visual.py --session_file {s_f} --height {h} --zoom {z} --epsilon {e} --categories '{c}'".format(s_f=args.session_file,h=args.height,z=args.zoom,e=args.epsilon,f_f=args.from_first,c=args.categories)
print(cmd)
os.system(cmd)

