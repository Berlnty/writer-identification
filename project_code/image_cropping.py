import cv2
import os
import numpy as np
from scipy import ndimage
from scipy.ndimage import label


def crop_image(image):
    height, width = image.shape[:2]



    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    # dilation
    kernel = np.ones((4, 4), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    labeled_img, num_connected_comp = label(img_dilation)
    objs = ndimage.find_objects(labeled_img)

    # list of tuples(x1,y1,x2,y2) each tuple for one connected component
    coorinates_bounding_rec = []

    for i in range(0, len(objs)):
        width_bounding_rec = (int(objs[i][1].stop) - 1) - (int(objs[i][1].start))
        height_bounding_rec = (int(objs[i][0].stop) - 1) - int(objs[i][0].start)
        if height_bounding_rec > 5:  # to exclude noise and points over letter
            coorinates_bounding_rec.append(
                (int(objs[i][1].start), int(objs[i][0].stop) - 1, int(objs[i][1].stop) - 1, int(objs[i][0].start)))
            cv2.rectangle(thresh, (coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][0],
                                   coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][1]), (
                          coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][2],
                          coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][3]), (255, 255, 255), 1)

    # sort according to x1
    sorted_acc_height= coorinates_bounding_rec.copy()
    coorinates_bounding_rec.sort(key=lambda tup: tup[3])


    coorinates_bounding_rec.sort(key=lambda tup: tup[2]-tup[0])
    line=[]
    line.append(coorinates_bounding_rec[len(coorinates_bounding_rec) - 1])
    line.append(coorinates_bounding_rec[len(coorinates_bounding_rec) - 2])
    line.append(coorinates_bounding_rec[len(coorinates_bounding_rec) - 3])

    line.sort(key=lambda tup: tup[3])

    index=[]
    index.append(0)
    index.append(0)
    for i, ctr in enumerate( sorted_acc_height):

        if(ctr == line[2]):index[1]=i
        elif(ctr == line[1]):index[0]=i

    _, _, _, ymin =sorted_acc_height[index[0]+1]
    _,ymax,_,_= sorted_acc_height[index[1]-1]

    get_new_width=[]
    for h in range(index[0]+1, index[1]):
        get_new_width.append(sorted_acc_height[h])

    get_new_width.sort(key=lambda tup: tup[0])
    xmin, _, _,_ = get_new_width[0]

    get_new_width.sort(key=lambda tup: tup[2])
    _, _, xmax, _ = get_new_width[len(get_new_width)-1]

    cropped= gray[ymin:ymax, xmin:xmax]
    #cv2.imshow('source', cv2.resize(image, (700, 1000)))
    #cv2.imshow('line', cv2.resize(cropped, (700, 1000)))
    #cv2.waitKey(0)
    return cropped


