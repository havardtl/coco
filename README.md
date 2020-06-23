# coco
Confocal counter (coco) is a program that extracts information about objects from confocal images. coco also contains a sister version for analyzing brightfield organoid images called boco. 

coco was designed to analyze confocal images of small intestinal organoids. Specifically, it is implemented to count the number of tuft cells per organoid as well as extracting size information about each organoid, but it should be useful for other types of quantification as well.

boco utilize many of the same functions as coco, but analyzes bright field images of organoids. Additionally, boco contains a neural network implementation to classify organoids into spheroids and organoids. 

All python scripts in main folder are runnable as command line programs. Run them with -h or --help option to get information on usage. 

Details of implementation and example of use can be found in the following paper. Please cite if you use this program. 
Lindholm et al. (June 2020). "Developmental pathways regulate cytokine-driven effector and feedback responses in the intestinal epithelium". bioRxiv. DOI/URL: https://doi.org/10.1101/2020.06.19.160747 

Use at your own risk. This repository is not actively maintained, but you are welcome to open an issue and I might look at it.  

## Installation
It is recommended to install dependencies with conda. You can utilize the environment file environment.yml to get a copy of my environment. Alternatively, I had success with python 3.6 using the following commands: 

`conda config --add channels conda-forge`

`conda install opencv pandas openpyxl xlrd reportlab`

I used .czi files from ZEISS for confocal image analysis. These where imported using the aicspylibczi library which can be installed with:

`pip install aicspylibczi`

Download repository: 

`git clone https://github.com/havardtl/coco.git`

Add folder with coco scripts to PATH by adding this line to your .bashrc: 

`export PATH="$PATH:/path/to/cooco/folder"`

Open the folder with images you want to analyze in bash and type the command you want to use. 

## FAQ
### What alternatives are there to this program?
ilastic can achieve a lot of the same image segmentation and classification as boco[1]. Macros in ImageJ can achieve most of the analysis except neural network based classification[2].

[1] Stuart Berg et al. (2019). "lastik: interactive machine learning for (bio)image analysis". Nature Methods. doi: https://doi.org/10.1038/s41592-019-0582-9 website: https://www.ilastik.org/

[2] Rueden C. et al. (2017), "ImageJ2: ImageJ for the next generation of scientific image data", BMC Bioinformatics, doi: https://doi.org/10.1186/s12859-017-1934-z

### How can I get imagej to open files windowless? 
1. Open FIJI2. Navigate to Plugins > Bio-Formats > Bio-Formats Plugins Configuration
3. Select Formats
4. Select your desired file format (e.g. “Zeiss CZI”) and select “Windowless”

## Todo: 
- Write simple test
- Remove code that tries to analyze stuff in 3D
- Put as much as possible of the code into functions with comments. 
- Simplify scripts into main commands "coco" and "boco" with sup-options
- Write jupiter notebook with example of usage. 
- Change default output of projections to add each object number
- Add possibility to add manual annotation of organoids and other annotations
- Add log file for each run so that you know settings used to segment. (git log -1 --format="%H" gives commit hash)



