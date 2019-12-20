import preprocessing
import load_paths
import convert
import detect_box
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--paths', help = "path to image")
ap.add_argument('-c', '--crop', help=" path save box")

args = vars(ap.parse_args())


imagePaths = list(paths.list_images(args['paths']))
crop_dir_path = args['crop']
detect_box.box_extraction(imagePaths, crop_dir_path)
preprocessing.preprocessing()
convert.get_text()

