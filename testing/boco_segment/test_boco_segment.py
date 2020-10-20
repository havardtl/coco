'''
This script runs boco_segment_intial.py and checks that the output is correct
'''

###################
# Imports
###################
import os
import shutil
import pandas as pd

###################
# Functions
###################

def delete_object(file_or_folder):
    #Delete an object if it exists
    if os.path.exists(file_or_folder):
        if os.path.isdir(file_or_folder):
            print("Deleting folder: "+file_or_folder)
            shutil.rmtree(file_or_folder)
        else: 
            print("Deleting file: "+file_or_folder)
            os.remove(file_or_folder)
    else: 
        print("Tried to delete object, did not exist. Path = "+file_or_folder)

def check_value(value_name,value,actual,checked_values,margin=0.0000001):
    '''
    Check whether a value is equal a value and report the result

    Params
    value_name     : str          : The name of the parameter that is checked
    value          : float        : Value that is being checked
    actual         : float        : The truth value to compare towards
    checked_values : list of bool : Result of test is appended to this list. True for passed test. 
    margin         : float        : margin of error for testing value. 
    '''
    if value <= (actual+margin) and value >= (actual-margin):
        passed_str = "OK"
        checked_values.append(True)
    else: 
        passed_str = "ERROR"
        checked_values.append(False)
    print(passed_str + "\t" + value_name+"\tvalue = "+str(value)+" actual = "+str(actual)+" margin = "+str(margin))
    
###################
# Reset folder
###################
delete_object("annotations")
delete_object("graphic_out_segment_raw")
delete_object("min_projections")
delete_object("ORGAI_session_file.txt")
delete_object("process_annotations.R")
delete_object("treatment_info.xlsx")

###################
# Run script
###################
cmd = "boco_segment_initial.py --debug"
print("Running command: "+cmd)
exit_val = os.system(cmd)
if not exit_val == 0: 
    raise ValueError("command did not exit properly: "+cmd)

###################
# Evaluate results
###################

checked_values = []

df = pd.read_csv("annotations/HL32_2_A03_d02.txt",sep=";",skiprows = 5)

print(df[["channel_id","center_x","center_y","area","equivalent_radius","annotation_this_channel_type","annotation_other_channel_type"]])      

check_value("n_objects",len(df.index),2,checked_values)
check_value("n_spheroid",sum(df["annotation_this_channel_type"]=="Spheroid"),1,checked_values)
check_value("n_budding",sum(df["annotation_this_channel_type"]=="Budding"),1,checked_values)

budding = df.loc[df["annotation_this_channel_type"]=="Budding"]
check_value("budding_eq_radius",float(budding["equivalent_radius"]),18.480977,checked_values,0.01)

spheroid = df.loc[df["annotation_this_channel_type"]=="Spheroid"]
check_value("spheroid_eq_radius",float(spheroid["equivalent_radius"]),32.854053,checked_values,0.01)









