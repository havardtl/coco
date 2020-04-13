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
- Add log file for each run so that you know settings used to segment. (git log -1 --format="%H" gives commit hash)


## dependencies
conda install opencv pandas openpyxl xlrd reportlab
pip install aicspylibczi

### To get imagej to open files windowless: 
1. Open FIJI2. Navigate to Plugins > Bio-Formats > Bio-Formats Plugins Configuration
3. Select Formats
4. Select your desired file format (e.g. “Zeiss CZI”) and select “Windowless”

### install python-bioformats
1. sudo apt-get install openjdk-8-jdk
2. export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64" (add to .bashrc)
3. conda install python=3.5 anaconda::javabridge cython
4. pip install python-bioformats
