import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import pickle
import random
from IPython.display import clear_output
from skimage.measure import shannon_entropy

parser = argparse.ArgumentParser(description='Segmentation, phan vung anh, tim phan ROI')
parser.add_argument('--train_output', default='', type=str, 
                    help='Duong dan luu du lieu training')
parser.add_argument('--test_output', default='', type=str, 
                    help='Duong dan luu du lieu testing')
parser.add_argument('--train_data', default='/', type=str, 
                    help='Duong dan den du lieu training')
parser.add_argument('--test_data', default='/', type=str, 
                    help='Duong dan den du lieu testing') 
args = parser.parse_args()


def findROICoord(img):
    h, w = img.shape
    x1 = 10000
    y1 = 10000
    x2 = 0
    y2 = 0
    for i in range(h):
        for j in range(w):
            if img[i, j] < 10:
                if i < y1:
                    y1 = i
                if j < x1:
                    x1 = j
                if i > y2:
                    y2 = i
                if j > x2:
                    x2 = j
    return x1, y1, x2, y2

def resize(img, require_size = (128, 128)):
    try:
        if img.shape < require_size:
            resized_img = cv2.resize(img, require_size, cv2.INTER_AREA)
        else:
            resized_img = cv2.resize(img, require_size, cv2.INTER_CUBIC)
    except Exception as e:
        print(str(e))
    return resized_img

def segment(data_path, SAVE_PATH):
    labels = [label for label in os.listdir(data_path)] #Duyet cac labels
    kernel = np.ones((3,3))
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)  
    for label in labels: #Duyet cac labels
        dir = os.path.join(data_path, label)
        for img_name in os.listdir(dir): #Duyet cac anh
            im_path = os.path.join(dir, img_name)
            img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE) #doc anh xam
            imgsave_name = '{}'.format(img_name) #Ten anh de luu            
            save_directory = os.path.join(SAVE_PATH, label)
            save_path = os.path.join(save_directory, imgsave_name)
            
            h, w = img.shape
            diff = 10
            for i in range(w): #duyet canh ben tren
                if img[0, i] > 220:
                    cv2.floodFill(img, None, (i, 0), 0, loDiff=diff, upDiff=diff)

            for i in range(w-1, -1, -1):  #duyet canh ben duoi
                if img[h - 1, i] > 220:
                    cv2.floodFill(img, None, (i, h-1), 0, loDiff=diff, upDiff=diff)
            
            img_org = copy.copy(img) #tao ban sao cua img
            
            _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #nhi phan hoa theo nguong OTSU
            img = 255 - img #invert anh
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #toan tu open
            img = cv2.dilate(img,kernel,iterations = 7) #toan tu gian no
            
            x1, y1, x2, y2 = findROICoord(img)
            x1 = np.maximum(x1 - 60, 0) #tranh tran khoi kich thuoc anh goc
            y1 = np.maximum(y1 - 60, 0)
            x2 = np.minimum(x2 + 60, w - 1)
            y2 = np.minimum(y2 + 60, h - 1)
            try:
                img_org = img_org[y1:y2, x1:x2]            
                if not os.path.exists(save_directory):
                    os.mkdir(save_directory)        
                img_org = resize(img_org, (128, 128))  
                entropy = shannon_entropy(img_org)
                if entropy <= 5.5:
                    print(entropy)
                    print(save_path)
                    continue
                cv2.imwrite(save_path, img_org)
#                 clear_output()
            except Exception as e:
                print(im_path, save_path, str(e))

segment(args.train_data, args.train_output)
segment(args.test_data, args.test_output)