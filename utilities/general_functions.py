import os
import pandas as pd
from io import StringIO

def load_session_file(path):
    '''
    Load session file linking minimal projections and their respective annotation data
    
    Params
    path    : str               : path to session file
    
    Returns
    df      : pandas.DataFrame  : df with all annotation data
    '''
    df = pd.read_csv(path)
    df.set_index("id",inplace=True)
    df["n"] = range(0,len(df.index))
         
    df = pd.read_csv(path)
    df.set_index("id",inplace=True)
    df["n"] = range(0,len(df.index))
    
    index = df.index[-1]
    for i in df.index:
        if not df.loc[i,'manually_reviewed']:
            index = i
            break
 
    return(df,index)

def load_annotation_file(path):
    '''
    Load annotation file in ORGAI specific format
    
    Params 
    path                : str               : file path to annotation file to load
    
    Params
    annotation          : pandas.DataFrame  : df with all annotation data
    reviewed_by_human   : bool              : If the file annotation has ever been reviewed manually, this is true
    next_org_id         : int               : the id of the next organoid to be added. To make sure that no organoid in this annotation ever gets the same id-number
    changelog           : str               : the file history
    
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
    
    if "next_org_id" in lines[1]:
        next_org_id = lines[1].split("=")[1]
        next_org_id = int(next_org_id.strip())

    changelog = lines[changelog_line+1:info_line]
    
    annotation_raw = ""
    for l in lines[info_line+1:]:
        annotation_raw = annotation_raw+l
    
    annotation_raw = StringIO(annotation_raw)

    annotation = pd.read_csv(annotation_raw)
    annotation['i'] = annotation['org_id']
    annotation.set_index('i',inplace=True,verify_integrity=True)
    return(annotation,reviewed_by_human,next_org_id,changelog)

def save_annotation_file(path,annotation,reviewed_by_human,next_org_id,changelog):
    '''
    Save annotation file in ORGAI specific format
    
    Params
    path                : str               : path to place for saving file
    annotation          : pandas.DataFrame  : df with all annotation data
    reviewed_by_human   : bool              : If the file annotation has ever been reviewed manually, this is true
    next_org_id         : int               : the id of the next organoid to be added. To make sure that no organoid in this annotation ever gets the same id-number
    changelog           : str               : the file history
    
    '''
    with open(path,'w') as f:
        if reviewed_by_human: 
            f.write("reviewed_by_human = True\n")
        else:
            f.write("reviewed_by_human = False\n")
        f.write("next_org_id = "+str(next_org_id)+"\n")
        f.write("--changelog--\n")
        for l in changelog: 
            f.write(l)
        f.write("--organoids--\n")
    annotation.to_csv(path,mode="a",index=False)


def walk_to_df(folder,id_split=None,filter_str=None):
    '''
    #inuse
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
                if filter_str is not None: 
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
    return(df)

def evos_to_imageid(name):
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
        
def find_stacks(folder):
    '''
    Find all stacks in folder and report them back as data frame with id column
    
    Params 
    folder : str : path to folder with all stacks
    
    Returns
    stacks : pandas.DataFrame : Data frame with path to stack and individual id of stack
    
    '''
    stacks = walk_to_df(folder,id_split=None,filter_str="TIF")
    if len(stacks) ==0:
        raise ValueError("Did not find any stacks")

    stacks_id = []
    stacks_path = []
    for i in stacks.index:
        stacks_id.append(evos_to_imageid(stacks.loc[i,"file"]))
        stacks_path.append(os.path.join(stacks.loc[i,"root"],stacks.loc[i,"file"]))
    stacks["id"] = stacks_id
    stacks["full_path"] = stacks_path
    
    return stacks