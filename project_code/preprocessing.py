import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import label
import math





def preprocess(img_gray):
    global_x = 0
    global_y = 20
    f_row = 3
    f_col = 3

    # Read the image you want connected components of
    #img_gray = cv2.imread('2.PNG', cv2.IMREAD_GRAYSCALE)

    ret, thresh = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)

    labeled_img, num_connected_comp = label(thresh)
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
    coorinates_bounding_rec.sort(key=lambda tup: tup[0])

    ############## get width of each bounding_rec ########################

    width_bound_recS = []
    for i in range(0, len(coorinates_bounding_rec)):
        width_bound_rec = coorinates_bounding_rec[i][2] - coorinates_bounding_rec[i][0]
        width_bound_recS.append(width_bound_rec)
    #####################################################################

    ############## get Centre of mass of each bounding_rec ########################
    X_centreOfMass_bound_recS = []
    Y_centreOfMass_bound_recS = []
    for i in range(0, len(coorinates_bounding_rec)):
        X_centre = coorinates_bounding_rec[i][2] - coorinates_bounding_rec[i][0]
        Y_centre = coorinates_bounding_rec[i][1] - coorinates_bounding_rec[i][3]
        X_centreOfMass_bound_recS.append(X_centre / 2)
        Y_centreOfMass_bound_recS.append(Y_centre / 2)
    #####################################################################

    ############## get Avg height of each bounding_rec ########################
    height_bound_recS = []
    for i in range(0, len(coorinates_bounding_rec)):
        height_bound_rec = coorinates_bounding_rec[i][1] - coorinates_bounding_rec[i][3]
        height_bound_recS.append(height_bound_rec)
    #####################################################################
    # Avg_height_bound_recS=sum(height_bound_recS)/len(height_bound_recS)

    last_x = 0
    last_y = 0

    sliced_images = []
    new_image = np.ones((f_row * 256, f_col * 128), np.int8)
    for i in range(0, len(coorinates_bounding_rec)):
        start_x = coorinates_bounding_rec[i][0]
        end_x = start_x + width_bound_recS[i] + 1
        start_y = coorinates_bounding_rec[i][3]
        end_y = start_y + height_bound_recS[i] + 1
        img_Slice = img_gray[start_y:end_y, start_x:end_x]
        sliced_images.append(img_gray[start_y:end_y, start_x:end_x])


    sum_h = 0
    no_image_in_row = 0
    for i in range(0, len(sliced_images)):

        offset_h, offset_w = sliced_images[i].shape[:2]
        if(offset_w > 384): sliced_images[i]= cv2.resize(sliced_images[i],(384,offset_h))
        if (offset_h > 768): sliced_images[i] = cv2.resize(sliced_images[i], (offset_w, 768))
       # print(global_x + offset_w )
        if (global_x + offset_w >= 384):

            global_y = global_y + ((sum_h / no_image_in_row)/2 )
            global_x = 0
            sum_h = 0
            no_image_in_row = 0

        temp_y = math.ceil(global_y - Y_centreOfMass_bound_recS[i])
        if(temp_y<0): temp_y=0
        sum_h += offset_h
        no_image_in_row += 1
        offset_y = temp_y + offset_h
        if (offset_y < (768)):
            new_image[temp_y: offset_y, global_x:global_x + offset_w] = sliced_images[i]
            global_x = global_x + offset_w

    features_pattern = []

    looping_rows= int(global_y/256)+1
    #new_image=new_image[0:(looping_rows*256),0:f_col * 128]

    for k in range(0, looping_rows):
        for i in range(0, f_col):
            features_pattern.append(new_image[k * 256:(k + 1) * 256, i * 128:(i + 1) * 128])

    #print('no of blocks: ',len(features_pattern))
    #cv2.imshow('block',new_image)
    #cv2.waitKey(0)
    return features_pattern
   # return new_image

#preprocess(cv2.imread('1_1.PNG', cv2.IMREAD_GRAYSCALE));