
class Roi_3d: 

    def __init__(self,id_roi_3d,contours,z_stack,channel_index):
        '''
        Region of interest in 3 dimensions. Built from 2D contours that overlapp.

        Params
        id_roi_3d     : int              : id of the roi_3d object
        contours      : list of Contours : Contours that specify 2D objects in image
        z_stack       : Z_stack          : z_stack these contours are from
        channel_index : int              : Channel index these contours are from
        '''
        self.id_roi_3d = id_roi_3d 
        self.contours = contours
        self.z_stack = z_stack
        self.channel_index = channel_index 

        self.is_inside_roi = None 

        self.data = None
    
    def is_inside_combined_roi(self,combined_rois):
        '''
        Each contour in a Roi_3d has information about which combined_mask contour it is inside. This function looks through Roi_3d objects and converts information about which contour it is inside to which combined_mask Roi_3d object it is inside. 
        
        Params
        combined_rois : list of Roi_3d : List of contours to check whether these contours are inside         
        '''
        is_inside_roi = [] 
        for c in self.contours:
            for c_i in c.is_inside:
                is_inside_roi.append(c_i.roi_3d_id)
        is_inside_roi = list(set(is_inside_roi))
        self.is_inside_roi = is_inside_roi

    def build(self):
        '''
        Build data about roi_3d object from the list of contours and other information
        '''

        data = {
            "id_roi_3d"    : self.id_roi_3d,
            "z_stack"      : self.z_stack.id_z_stack,
            "x_res"        : self.z_stack.physical_size["x"],
            "y_res"        : self.z_stack.physical_size["y"],
            "z_res"        : self.z_stack.physical_size["z"],
            "res_unit"     : self.z_stack.physical_size["unit"], 
            "file_id"      : self.z_stack.file_id, 
            "series_index" : self.z_stack.series_index,  
            "time_index"   : self.z_stack.time_index,
            "channel_index": self.channel_index,
            "is_inside"    : None,
            "is_inside_roi": None,
            "volume"       : 0,
            "n_contours"   : len(self.contours),
            "contour_ids"  : None,
            "contours_centers_xyz" : None,
            "biggest_xy_area":None,
            "img_dim_y" : self.z_stack.img_dim[0],
            "img_dim_x" : self.z_stack.img_dim[1],
            "mean_x_pos" : None,
            "mean_y_pos" : None,
            "mean_z_index" : None, 
            "at_edge"      : None,
            "annotation_this_channel_type" : None,
            "annotation_this_channel_id" : None,
            "annotation_other_channel_type" : None,
            "annotation_other_channel_id" : None,
            "manually_reviewed" : None 
        }

        to_unit = data["x_res"] * data["y_res"] * data["z_res"]

        mean_grey_channels = [s for s in self.contours[0].data.keys() if "sum_grey_C" in s]
        sum_pos_pixels_channels = [s for s in self.contours[0].data.keys() if "sum_positive_C" in s]
        
        temp = {} 
        for tag in mean_grey_channels+sum_pos_pixels_channels: 
            temp.update({tag:0})
         
        for c in self.contours: 
            for tag in mean_grey_channels+sum_pos_pixels_channels:
                temp[tag] = temp[tag] + (c.data[tag][0] *to_unit)
        
        data.update(temp)
        if self.is_inside_roi is not None:  
            for inside_roi in self.is_inside_roi: 
                if data["is_inside_roi"] is None: 
                    data["is_inside_roi"] = str(inside_roi)
                else: 
                    data["is_inside_roi"] = data["is_inside_roi"] +"," +str(inside_roi)
        
        sum_x_pos = 0
        sum_y_pos = 0
        sum_z_pos = 0
        annotation_this_channel_type = []
        annotation_this_channel_id = []
        annotation_other_channel_type = []
        annotation_other_channel_id = []

        for c in self.contours:
            data["volume"] = data["volume"] + (c.data["area"]*to_unit)
            
            if data["contour_ids"] is None: 
                data["contour_ids"] = str(c.id_contour)
            else:
                data["contour_ids"] = data["contour_ids"] +","+str(c.id_contour)
            
            center = "x"+str(c.data["centroid_x"])+"y"+str(c.data["centroid_y"])+"z"+str(c.data["z_index"])
            if data["contours_centers_xyz"] is None: 
                data["contours_centers_xyz"] = center 
            else:
                data["contours_centers_xyz"] = data["contours_centers_xyz"]+","+center
           
            if data["biggest_xy_area"] is None: 
                data["biggest_xy_area"] = c.data["area"]
            elif data["biggest_xy_area"] < c.data["area"]:
                data["biggest_xy_area"] = c.data["area"]

            sum_x_pos += int(c.data["centroid_x"])
            sum_y_pos += int(c.data["centroid_y"])
            sum_z_pos += int(c.data["z_index"])

            if data["at_edge"] is None:
                    data["at_edge"] = c.data["at_edge"] 
            elif c.data["at_edge"] is not None:
                if c.data["at_edge"]: 
                    data["at_edge"] = c.data["at_edge"]
            
            if data["is_inside"] is None:
                data["is_inside"] = str(c.data["is_inside"])
            else:
                data["is_inside"] = data["is_inside"] +","+str(c.data["is_inside"])

            annotation_this_channel_type  = annotation_this_channel_type  + c.annotation_this_channel_type
            annotation_this_channel_id    = annotation_this_channel_id    + c.annotation_this_channel_id 
            annotation_other_channel_type = annotation_other_channel_type + c.annotation_other_channel_type
            annotation_other_channel_id   = annotation_other_channel_id   + c.annotation_other_channel_id 

            if data["manually_reviewed"] is None: 
                data["manually_reviewed"] = c.manually_reviewed
            elif not data["manually_reviewed"]: 
                data["manually_reviewed"] = c.manually_reviewed
        
        annotation_this_channel_type_set,  annotation_this_channel_id_set  = set_zipped_list(annotation_this_channel_type, annotation_this_channel_id )
        annotation_other_channel_type_set, annotation_other_channel_id_set = set_zipped_list(annotation_other_channel_type,annotation_other_channel_id)

        data["annotation_this_channel_type"]  = ",".join(annotation_this_channel_type_set)
        data["annotation_this_channel_id"]    = ",".join(annotation_this_channel_id_set)
        data["annotation_other_channel_type"] = ",".join(annotation_other_channel_type_set)
        data["annotation_other_channel_id"]   = ",".join(annotation_other_channel_id_set)

        if data["n_contours"]>0:
            data["mean_x_pos"]   = sum_x_pos/data["n_contours"]
            data["mean_y_pos"]   = sum_y_pos/data["n_contours"]
            data["mean_z_index"] = sum_z_pos/data["n_contours"]

        self.data = data

    def __repr__(self):
        string = "{class_str} id: {class_id} built from n contours: {n}".format(class_str = self.__class__.__name__,class_id = self.id_roi_3d,n = len(self.contours))
        return string
