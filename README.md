# coco
Confocal counter (coco) that extracts information about objects from confocal images

## Todo: 
- Add perimeter calculation to roi_3d objects (based on cylinder surface equation)
- Add other measurments to roi_3d objects
- Make sure units are consistent between measurements
- Make the program file name agnostic (information should be added in annotation.xlsx)
- Write a good readme
- Add output with projection and each object number
- Add possibility to add manual annotation of organoids and other annotations
- Reduce amount of repeated code in each file run in command line
- Add log file for each run so that you know settings used to segment. (git log -1 --format="%H" gives commit hash)
