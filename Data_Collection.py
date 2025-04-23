import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap= cv2.VideoCapture(0)     # create a object of image at camera id =0

detector =HandDetector(maxHands=2)
offset=20
imgSize=300
counter=0
folder="Data/Z"
if not os.path.exists(folder):
    os.makedirs(folder)
while True:
    success,img =cap.read()      #read function inside cap class will start capturing frame
                                # a boolean value indecating if it is successfully captured or not =success
                                # captured image itself =img
    hands,img=detector.findHands(img)    #take image as argument, and give two output
                                            #hand vector containing all the hands indexed from 0 ,hands[0],hands[1].
                                            #img vector containing the image of hands

    if hands:
        hand =hands[0]
        x,y,w,h=hand['bbox']
        # w=width of the box
        # h= height of the box
        # x,y= x and y co-ordinate of the box
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape=imgCrop.shape


        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal =math.ceil(k+w)

            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((300-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k + h)

            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300 - hCal) / 2)
            imgWhite[hGap:hCal+hGap,:] = imgResize


        cv2.imshow("Frame",img)
        cv2.imshow("white", imgWhite)


    #cv2.imshow("Image",img)    #  it will show the image in the console

    key = cv2.waitKey(1)  # moved this line outside of the if hands block
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Z_{time.time()}.jpg', imgWhite)
        print(counter)

