'''
This script runs coco_segment_2D.py and checks that the output is correct
'''

########################
# Imports
########################
import os
import shutil
import pandas as pd 

########################
# Functions
########################
def delete_folder(folder):
    #Delete a folder if it exists
    if os.path.exists(folder):
        print("Deleting folder: "+folder)
        shutil.rmtree(folder)
    else: 
        print("Tried to delete folder, did not exist. Folder = "+folder)

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
    
########################
# Reset folder
########################

delete_folder("graphical")
delete_folder("stats")
delete_folder("raw/temp")

########################
# Run script
########################
cmd = "coco_segment_2D.py --debug"
print("Running command: "+cmd)
exit_val = os.system(cmd)
if not exit_val == 0: 
    raise ValueError("command did not exit properly: "+cmd)
    
########################
# Evaulate results
########################
stats_folder = "stats/2D_contours_stats"
stats_files = os.listdir(stats_folder)

df_list = []
for s in stats_files:
    stats_file_path = os.path.join(stats_folder,s)
    print(stats_file_path)
    temp = pd.read_csv(stats_file_path,sep=";")
    df_list.append(temp)

df = pd.concat(df_list)
print(df)

print("\nObjects in all channels: ")
channels = df["channel_id"].unique()
for c in channels: 
    this_channel = df.loc[df["channel_id"]==c,]
    this_channel = this_channel[["channel_id","area","equivalent_radius","annotation_this_channel_type","annotation_other_channel_type"]]
    print(this_channel)

checked_values = []
print("")
C0 = df.loc[df["channel_id"]=="C0"]
check_value("n_objects_in_C0",len(C0.index),4,checked_values)
check_value("tuft_in_C0",sum(C0["annotation_this_channel_type"]=="Tuft_cell"),4,checked_values)
check_value("biggest_eq_radius_C0",C0["equivalent_radius"].max(),2,checked_values,0.25)
check_value("smallest_eq_radius_C0",C0["equivalent_radius"].min(),2,checked_values,0.25)

C1 = df.loc[df["channel_id"]=="C1"]
check_value("n_objects_in_C1",len(C1.index),4,checked_values)
check_value("biggest_eq_radius_C1",C1["equivalent_radius"].max(),35,checked_values,0.5)
check_value("smallest_eq_radius_C1",C1["equivalent_radius"].min(),18,checked_values,0.5)
check_value("median_eq_radius_C1",C1["equivalent_radius"].median(),25,checked_values,0.5)

C2 = df.loc[df["channel_id"]=="C2"]
check_value("n_objects_in_C2",len(C2.index),3,checked_values)
check_value("biggest_eq_radius_C2",C2["equivalent_radius"].max(),35,checked_values,0.5)
check_value("smallest_eq_radius_C2",C2["equivalent_radius"].min(),18,checked_values,0.5)
check_value("median_eq_radius_C2",C2["equivalent_radius"].median(),25,checked_values,0.5)

C3 = df.loc[df["channel_id"]=="C3"]
check_value("n_objects_in_C3",len(C3.index),2,checked_values)
check_value("Goblet_in_C3",sum(C3["annotation_this_channel_type"]=="Goblet_cell"),2,checked_values)
check_value("max_eq_radius_C3",C3["equivalent_radius"].max(),5,checked_values,0.5)
check_value("min_eq_radius_C3",C3["equivalent_radius"].min(),5,checked_values,0.5)

Ccomb = df.loc[df["channel_id"]=="Ccomb"]
check_value("n_objects_in_Ccomb",len(Ccomb.index),4,checked_values)
check_value("budding_in_Ccomb",sum(Ccomb["annotation_this_channel_type"]=="Budding"),1,checked_values)
check_value("spheroid_in_Ccomb",sum(Ccomb["annotation_this_channel_type"]=="Spheroid"),3,checked_values)
check_value("biggest_eq_radius_Ccomb",Ccomb["equivalent_radius"].max(),25,checked_values,0.5)
check_value("smallest_eq_radius_Ccomb",Ccomb["equivalent_radius"].min(),18,checked_values,0.5)

#TODO: adds checks for signal from other channel

budding = Ccomb.loc[Ccomb["annotation_this_channel_type"].str.contains("Budding")]
check_value("budding_size",float(budding["equivalent_radius"]),25,checked_values,0.5)
check_value("budding_Tufts",float(budding["annotation_other_channel_type"].str.count("Tuft_cell")),4,checked_values,0.5)
check_value("budding_Goblets",float(budding["annotation_other_channel_type"].str.count("Goblet_cell")),2,checked_values,0.5)

x_res = df["x_res"].unique()
y_res = df["y_res"].unique()
z_res = df["z_res"].unique()

check_value("x_res",x_res,1.38,checked_values,0.01)
check_value("y_res",y_res,1.38,checked_values,0.01)
check_value("z_res",z_res,11,checked_values)

print("Values passing test: "+str(sum(checked_values))+"\tTotal values: "+str(len(checked_values)))
if sum(checked_values) != len(checked_values): 
    raise ValueError("Some values did not pass test! Check output above for details")



