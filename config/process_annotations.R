###############################
# Dependencies
################################
library(openxlsx)
library(reshape2)
library(dplyr)

main_path <- dirname(rstudioapi::getSourceEditorContext()$path)

annotations_path <- file.path(main_path,"annotations")
meta_data_file <- file.path(main_path,"treatment_info.xlsx")

annotation_corrected_path <- file.path(main_path,"segmented_post_manual","annotations")

projections_raw_dir <- file.path(main_path,"min_projections")
projections_pdf <- file.path(main_path,"min_projections_pdf")
projections_cropped_dir <- file.path(projections_pdf,"min_projections_cropped")

save_path <- file.path(main_path,"annotations_summarized.xlsx")

plot_images <- FALSE

area_thresh <- 300

####################################################
# Check if manual correction is performed, and 
# swap to that if it is done.
####################################################
if(dir.exists(annotation_corrected_path)){
  annotations_path <- annotation_corrected_path 
  save_path <- file.path(main_path,"segmented_post_manual","annotations_summarized_postsegment.xlsx")
}

####################################################
# get metadata, if available
####################################################
meta_data_present <- file.exists(meta_data_file)

if(meta_data_present){
  temp <- read.xlsx(meta_data_file)  
  if (!is.null(temp)){
    meta_data <- temp 
  }else{
    meta_data_present <- FALSE
  }
}

###############################
# Read annotations
################################
annotation_files <- list.files(annotations_path)
df <- data.frame()
f <- annotation_files[1]
for(f in annotation_files){
  temp <- readLines(file.path(annotations_path,f))
  
  index<-grep("--organoids--",temp)
  temp<-temp[(index+1):length(temp)]
  temp<-temp[nchar(temp)>0]
  
  temp <- read.table(text = temp,sep=";",as.is = T,header=1)
  temp$id <- tools::file_path_sans_ext(f)
  df <- rbind(df,temp)
}


temp <- colsplit(df$id,"_d",c("well_id","day"))
df$well_id <- temp$well_id
df$day <- temp$day

###############################
# Group
################################
df$day_char <- paste0("Day ",df$day)

df <- filter(df,!grepl("Junk",type,ignore.case = T))

#Do not calculate stats for to small objects
#Spheroids and budding fractions are not filtered out. 
v <- c("area","equivalent_radius","perimeter","circularity","hull_tot_area","radius_enclosing_circle")
df[df$area<area_thresh | is.na(df$area),v] <- NA

df_by_well <- df %>%
  group_by(well_id,day) %>%
  summarize(day_char = day_char[1],
            n_w_stats = sum(!is.na(area)),
            meanArea = mean(area,na.rm=T),
            sdArea = sd(area,na.rm=T),
            meanEqRadius = mean(equivalent_radius,na.rm=T),
            sdEqRadius = sd(equivalent_radius,na.rm=T),
            meanPerimeter = mean(perimeter,na.rm=T),
            sdPerimeter = sd(perimeter,na.rm=T),
            meanCirculairity = mean(circularity,na.rm=T),
            sdCircularity = sd(circularity,na.rm=T),
            tot_annotated=n(),
            spheroids = sum(type=="Spheroid",na.rm=T),
            budding = sum(type=="Budding",na.rm=T),
            exploded = sum(type=="Exploded",na.rm=T))

if(meta_data_present){
  df_by_well <- left_join(df_by_well,meta_data,by="well_id")
}

write.xlsx(df_by_well,save_path)

#####################################################################
# plot images
#####################################################################

if(plot_images){
  library(magick)
  library(ggimage)
  library(grid)
  
  dir.create(projections_pdf,showWarnings = F)
  
  # If not exists, create cropped jpgs from raw min projections
  projections_raw <- list.files(projections_raw_dir,recursive = T)
  projections_cropped <- projections_raw
  
  create_jpgs = FALSE
  if(!file.exists(projections_cropped_dir)){
    create_jpgs = TRUE
    print(paste("The jpg folder does not exist, creating all jpgs in: ",projections_cropped_dir))
    dir.create(projections_cropped_dir,showWarnings = F)
  }else{
    print("jpg folder exists, not creating jpgs")
  }
  
  i<-5
  if(create_jpgs){
    for(i in 1:length(projections_raw)){
      raw_img <- image_read(file.path(projections_raw_dir,projections_raw[i]))
      jpg_path <- file.path(projections_cropped_dir,projections_cropped[i])
      raw_img <- image_crop(raw_img,"1000x1000+164+24")
      image_write(raw_img,path=jpg_path,format="jpg")
    }
  }

  # Prepare image plotting
  projections <- data.frame(file_path=list.files(projections_cropped_dir,recursive = T),stringsAsFactors = F)
  projections$id <- gsub(".png","",projections$file_path)
  projections$file_path <- file.path(projections_cropped_dir,projections$file_path)
  
  temp <- colsplit(projections$id,"_d",c("well_id","day"))
  projections <- cbind(projections,temp)
  
  if(meta_data_present){
    projections <- left_join(projections,meta_data,by="well_id")
  }else{
    projections$treatment <- "" 
  }
  
  projections$treatment <- factor(projections$treatment)
  
  #Plot images
  for( i in 1){
    p <- ggplot(projections,aes(NA,NA)) +facet_grid(treatment+well_id~paste("day",day)) + geom_image(aes(image=file_path),size=1) + 
      theme(plot.margin=unit(c(30,5.5,5.5,5.5),"points"),axis.title = element_blank(),axis.text = element_blank(),axis.ticks = element_blank()) 
    p
    
    out_width = length(unique(projections$day)) *1.5+0.64
    out_height = length(unique(projections$well_id))*1.5+1
    ggsave(file.path(projections_pdf,paste0("all_wells.pdf")),width=out_width,height=out_height,units="in",dpi=600,limitsize = F)
  }
  
  one_per_treatment <- projections[order(projections$treatment,projections$id),]
  one_per_treatment <- projections[!duplicated(paste(projections$treatment,projections$day)),]
  
  if(meta_data_present){
    for( i in 1){
      p <- ggplot(one_per_treatment,aes(NA,NA)) +facet_grid(paste("day",day)~treatment+well_id) + geom_image(aes(image=file_path),size=1) + 
        theme(plot.margin=unit(c(30,5.5,5.5,5.5),"points"),axis.title = element_blank(),axis.text = element_blank(),axis.ticks = element_blank()) 
      p
      
      out_height = length(unique(one_per_treatment$day)) *1.5+0.64+0.3
      out_width = length(unique(one_per_treatment$well_id))*1.5+1
      ggsave(file.path(projections_pdf,paste0("one_per_treatment.pdf")),width=out_width,height=out_height,units="in",dpi=600,limitsize=F)
    }
  }
  
}

