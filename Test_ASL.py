import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Adding classification model 
classifier = Classifier("keras_model.h5", "labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "Calm down", "D", "E", "F", "G", "H", "Hello", "I", "I hate you", "I love you", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "Stop", "T", "U", "V", "W", "X", "Y"]

while True:
    sucess, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]                     # hands[0] means only single hand
        

        # bounded box information of hand    
        # w, h is width and height of image
        x, y, w, h = hand['bbox']   
           

        # Creating a image with white back-ground
        # Creating a matrix of ones of size 300 * 300 * 3
        # Color value range from 0 to 255 i.e., 8 bit value 
        # uint is unsigned integer of 8 bits
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255        

        # y is starting height, y+h is ending heigth & x is starting width, x+w is ending width
        # offset value is added to increase the boundry of cropped image
        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]   

        # placing the cropped image on the white image
        # Placing image crop matrix inside the image white matrix
        imageCropShape = imgCrop.shape

        aspecRatio = h/w 
        # Resizing an image if the height is max
        # In this case, height is fixed 
        if aspecRatio > 1:
            k = imgSize / h                                 # k is constant
            wCal = math.ceil(k * w)                         # wCal is calculated width
            imgResize = cv.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)          # To make the image at the center
            imgWhite[:, wGap : wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            #print(prediction, index)

        # In this case, width is fixed
        else:
            k = imgSize / w                                 # k is constant
            hCal = math.ceil(k * h)                         # hCal is calculated height
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)          # To make the image at the center
            imgWhite[hGap : hCal + hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        # Creating a rectangle around written output
        cv.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 225, y - offset), (255, 4, 255), cv.FILLED)

        cv.putText(imgOutput, labels[index], (x + 20, y - 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Creating a rectangle around the hand
        cv.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset , y + h + offset), (252, 4, 252), 4)

        #cv.imshow("ImageCrop",imgCrop)
        #cv.imshow("ImageWhite",imgWhite)

    cv.imshow('Image', imgOutput)
    cv.waitKey(1)    