setBatchMode(true);

arguments = getArgument()

arguments_split = split(arguments,",")
in_path = arguments_split[0]
out_path = arguments_split[1]

print(arguments)
print("in_path: "+in_path)
print("out_path: "+out_path)

open(in_path);
run("OME-TIFF...", "save="+out_path+" write_each_z_section write_each_timepoint write_each_channel compression=Uncompressed");
print("Done");
run("Quit");