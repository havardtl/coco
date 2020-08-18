import coco_package.AI_functions
import coco_package.image_processing
import coco_package.info
import coco_package.make_pdf
import coco_package.raw_image_read
import coco_package.visual_editor

def set_verbose():
    AI_functions.set_verbose()
    image_processing.set_verbose()
    info.set_verbose()
    make_pdf.set_verbose()
    raw_image_read.set_verbose()
