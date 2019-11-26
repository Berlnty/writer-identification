import os
import cv2
import image_cropping
#import cropping
#import preprocessing
import svm_knn
import numpy as np
import time
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from skimage import data
import re

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-1])
def crop2(image):
    ret, thresh = cv2.threshold(image, 150, 255, 0)
    im2, contours, hierarchy = cv2.findContours(255 - thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(self.lines[i], contours, -1, (155, 155, 155), 15)
    miny = 10000
    maxy = 0
    minx = 10000
    maxx = 0
    for c in contours:
        if cv2.contourArea(c) < 23:  # kant 2 , b2et 23!
            continue
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if y < miny:
            miny = y
        if y + h > maxy:
            maxy = y + h
        if x < minx:
            minx = x
        if x + w > maxx:
            maxx = x + w
    image = image[miny:maxy, minx:maxx]
    return image
def main():
    hight=int(2334/1)
    #read the data
    vote=0
    num=0
    f=open("Result.txt","w")
    tf = open("Time.txt", "w")

    extension = '.JPG'                                              #extension of yhe image
    path = 'G:\\college\\4th_year\\Pattern_Recognition\\Project\\handwritting\\submitted_code\\data'         #the path of folder data

    directory = os.listdir(path)

    directory . sort(key=sortKeyFunc)
   # directory=sorted(directory,key=lambda x: int(os.path))

    for foldername in directory:
        timebefore = time.time()
        training_features = []
        labels = []
        for t in range(1, 4):
            for i in range(1, 3):
                training_image = cv2.imread(path + '\\' + foldername + '\\' + str(t) + '\\' + str(i) + extension)
                #GUSSIAN FILTER
                kernel = np.ones((5, 5), np.float32) / 25
                training_image = image_cropping.crop_image(training_image)

                #print("after cropping",training_image.shape)
                training_image = cv2.filter2D(training_image, -1, kernel)

                #GRAY LEVEL and croping

                #training_image=cv2.cvtColor(training_image,cv2.COLOR_RGB2GRAY)
                labels.append(t)
                print(foldername+'\\'+ str(t)+'\\'+str(i))
                lbp2 = local_binary_pattern(training_image, 16, 1, method='default')
                (hist, _) = np.histogram(lbp2[np.where(lbp2<175)],density=True,bins=256,range=(0, 256))
                eps = 1e-7
                # normalize the histogram
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)

                training_features.append(hist)
        test_image = cv2.imread(path + '\\' + foldername + '\\' + 'test' + extension)
        #GUSSIAN FILTER
        kernel = np.ones((5, 5), np.float32) / 25
        test_image = image_cropping.crop_image(test_image)
        test_image = cv2.filter2D(test_image, -1, kernel)
        # GRAY LEVEL and croping

        #test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        #lbp for the testing set :V
        lbp2 = local_binary_pattern(test_image, 16, 1, method='default')
        (hist, _) = np.histogram(lbp2[np.where(lbp2<175)],density=True,bins=256,range=(0, 256))
        eps = 1e-7
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        test_features = []
        test_features.append(hist)
        #send data to knn

        result = svm_knn.myknn(training_features, labels, test_features)
        print("folder ",foldername)
        print(result)
        tf.write(str(time.time()-timebefore))
        tf.write('\n')
        f.write(str(result))
        f.write('\n')
    tf.close()
    f.close()

if __name__== "__main__":
    main()