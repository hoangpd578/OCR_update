import cv2
import os
from imutils import paths
import argparse
# load image path from the foder bat ki
def load(imagePaths):
    datas = []
    label_foders = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2. imread(imagePath)
        label_foder = imagePath.split(os.path.sep)[-1]

        # Xu ly label
        label_foder = label_foder.split(".")
        label_foder = label_foder[0]
        datas.append(image)
        label_foders.append(label_foder)

    print(label_foders)
    return datas, label_foders