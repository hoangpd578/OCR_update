import tempfile
import cv2
from PIL import Image
import numpy as np
from imutils import paths
import logging
import argparse
import load_paths
# Handling nhung box in foder crop_image
def preprocessing():
    impaths = list(paths.list_images("../DetectText_OCR/crop_image/"))

    for impath in impaths:
        img = Image.open(impath)
        lx, wy = img.size
        factor = max(1, int(1800/ lx))
        size = factor *lx, factor*wy

        img_resize = img.resize(size, Image.ANTIALIAS)
        temp_file = tempfile.NamedTemporaryFile(delete= False, suffix= '.jpg')
        temp_filename = temp_file.name
        img_resize.save(temp_filename, dpi = (300, 300))

        #remove noise

        img = cv2.imread(temp_filename, 0)
        fillter = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
        kernel = np.ones((1, 1), np.uint8)

        opening = cv2.morphologyEx(fillter, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        or_image = cv2.bitwise_or(th3, closing)

        #Handling_path save
        path_save = impath[: len(impath) - 4]
        path_save.replace("crop_image", "image_hanlding")

        cv2.imwrite(path_save + "_handling" + ".jpg", or_image)
        print(path_save)



