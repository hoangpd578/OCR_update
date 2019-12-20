import cv2
import pytesseract
from imutils import paths
from pytesseract import Output
import os

# Covert boxs to file txt
def get_text():
    foder = os.listdir("../DetectText_OCR/crop_image/")
    for i in range(len(foder)):
        k = "../DetectText_OCR/crop_image/" + foder[i]
        paths_ = list(paths.list_images(k))
        text_ = []
        for path in paths_:
            img = cv2.imread(path,0)
            text = pytesseract.image_to_string(img, lang= 'vie', config = '-l vie --oem 1 --psm 7')
            text_.append(text + "\n")
        with open("result/" + str(foder[i]) + ".txt", 'w') as myFile:
            myFile.write(" ".join(text_))