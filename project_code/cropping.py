import cv2
import numpy as np


def crop_image(image):
    height, width = image.shape[:2]
    image2=image.copy()
    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    cv2.imshow('rode',cv2.resize(img_dilation,(700,1000)))
    cv2.waitKey(0)

    # find contours
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

    cropping_lines = []  #supposed to be 3 lines
    mattering_ctrs=[]
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
        if ((w>0.02*width) and (h>0.02*height) and (w > h)):
            mattering_ctrs.append(ctr)
           # cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)

        # show ROI
        # cv2.imshow('segment no:'+str(i),roi)
        if ((h <= 0.05 * height) and (w > 0.6 * width)and (w>= 30*h)):
            cv2.rectangle(image2, (x, y), (x + w, y + h), (90, 0, 255), 2)
            cropping_lines.append(i)

    

    if(len(cropping_lines)==3):
        _, yline,_, _= cv2.boundingRect(sorted_ctrs[cropping_lines[2]])
        _, ymin, _, _ = cv2.boundingRect(sorted_ctrs[cropping_lines[1] + 2])
        _, ymax, _, h = cv2.boundingRect(mattering_ctrs[len(mattering_ctrs)- 1])
        ymax= ymax+h
        back=2
        while(ymax> yline):
            _, ymax, _, h = cv2.boundingRect(mattering_ctrs[len(mattering_ctrs) - back])
            back+=1
            ymax = ymax + h
    else:
        _, ymin, _, _ = cv2.boundingRect(sorted_ctrs[cropping_lines[1] + 2])
        _, ymax, _, h = cv2.boundingRect(mattering_ctrs[len(mattering_ctrs)- 2])
        ymax = ymax + h



    sorted_xaxis= sorted(mattering_ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    xmin,_,_,_=cv2.boundingRect(sorted_xaxis[0])

    sorted_weight=sorted(mattering_ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0]+ cv2.boundingRect(ctr)[2])
    xmax,_,w,_=cv2.boundingRect(sorted_weight[len(sorted_weight)-1])
    xmax= xmax+w



    cropped = gray[ymin:ymax, xmin:xmax]

    return cropped

