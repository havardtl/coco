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

`conda install opencv pandas openpyxl xlrd reportlab matplotlib tensorflow tk pillow`

`conda install keras`

I used .czi files from ZEISS for confocal image analysis. These where imported using the aicspylibczi library which can be installed with:

`pip install aicspylibczi`

Download repository: 

`git clone https://github.com/havardtl/coco.git`

Add folder with coco scripts to PATH by adding this line to your .bashrc: 

`export PATH="$PATH:/path/to/cooco/folder"`

Open the folder with images you want to analyze in bash and type the command you want to use. 



