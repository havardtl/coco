# coco
Confocal counter (coco) is a program that extracts information about objects from confocal images. coco also contains a sister version for analyzing brightfield organoid images called boco. 

coco was designed to analyze confocal images of small intestinal organoids. Specifically, it is implemented to count the number of tuft cells per organoid as well as extracting size information about each organoid, but it should be useful for other types of quantification as well.

boco utilize many of the same functions as coco, but analyzes bright field images of organoids. Additionally, boco contains a neural network implementation to classify organoids into spheroids and organoids. 

All python scripts in main folder are runnable as command line programs. Run them with -h or --help option to get information on usage. 

Details of implementation and example of use can be found in the following paper. Please cite if you use this program. 
Lindholm et al. (June 2020). "Developmental pathways regulate cytokine-driven effector and feedback responses in the intestinal epithelium". bioRxiv. DOI/URL: https://doi.org/10.1101/2020.06.19.160747 

## Installation
It is recommended to install dependencies with conda. I had success with python 3.8 using the following commands: 

`conda config --add channels conda-forge`

`conda install opencv pandas openpyxl xlrd reportlab matplotlib`

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

## Todo: 
- Write jupyter notebook with example of usage. 
- Add log file for each run so that you know settings used to segment. (git log -1 --format="%H" gives commit hash)



