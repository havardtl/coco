setBatchMode(true);

arguments = getArgument();

arguments_split = split(arguments,",");
in_path = arguments_split[0];
out_path = arguments_split[1];

print(arguments);
print("in_path: "+in_path);
print("out_path: "+out_path);

open(in_path);
run("Stack to Images");

wait(1000);

n_images = nImages();

for(i=1;i<=n_images;i++){
    selectImage(i);
    img_name = getTitle();
    
    print("i = "+i+"  "+img_name);
    
    img_name_vec = split(img_name," ");
    channel = img_name_vec[0];
    channel = split(channel,":/");
    channel = channel[1];
    channel = parseInt(channel);
    channel = channel -1;
    z_id = img_name_vec[1];
    z_id = split(z_id,":/");
    z_id = z_id[1];
    z_id = parseInt(z_id);
    z_id = z_id -1;
    
    out_name =  out_path + "_INFO_zid-"+z_id+"-channel-"+channel+"-_OTHER.ome.tiff";
    
    print(out_name);
    
    run("OME-TIFF...", "save="+out_name+" write_each_z_section write_each_timepoint write_each_channel compression=Uncompressed");
}
print("Done");
run("Quit");