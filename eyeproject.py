import cv2,sys
import numpy as np
import math
import matplotlib.pyplot as plt

def createHistorgramOfEyeColor(img, centers, eyeRadius, test=False) :
        # covert image to gray-scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert image to float so to create a mask
        gray = gray/255.0
        for circle in centers:
            center = (circle[0], circle[1])
            # get all pixels inside eye radius
            cv2.circle(gray, center, eyeRadius,2,-1)
        if test:
            cv2.imshow('test',gray)
        eyePixels = np.where(gray == 2)
        # get histogram of colors for eye_pixlels
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v  = cv2.split(hsv_image)
        eye_pixel_hue = h[eyePixels]
        # h[0] / h[180] = red, h[60] = green, h[120] = blue
        histogram = np.histogram(eye_pixel_hue, np.arange(180))
        return histogram, eyePixels, eye_pixel_hue, h , s, v 

def changeHueOfHistogram(histogram, eyePixels, eye_pixel_hue, h, s, v, color, test = False) :
    largestBins = findLargestNConsecutiveBins(histogram,30)
    if test:
        print(histogram)
        print(largestBins)
    if color == "brown" :
        destinationHue = np.concatenate((np.arange(170,180), np.arange(0,20)))
    elif color == "blue" :
        destinationHue = np.arange(100,131)
    elif color == "green" :
        destinationHue = np.arange(40,71)
    # loop through the largest bin of hue colors, and map them to the destination color
    for i,bin in enumerate(largestBins):
        eye_pixel_hue[eye_pixel_hue == bin] = destinationHue[i]
    # update hue of eye pixel region
    h[eyePixels] = eye_pixel_hue
    # recreate hsv image with updated hues, covert back to BGR and display
    newImage = cv2.merge([h,s,v])
    newImage = cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR)
    # cv2.imshow('new',newImage)
    return newImage

# finds the largest n consecutive numbers in an array when array is regarded as circular
def findLargestNConsecutiveBins(histogram,n) :
    bins = histogram[0]
    maxSum = 0
    maxBins = []
    lengthOfBins = len(bins)
    for i in range(0, lengthOfBins):
        # max numbers can start at the end of the array and wrap to the beggining so we need to account for those
        # ex: [44, 2, 3, 40, 42] -> we need two arrays [0] & [3:4]
        overflow = i + n > lengthOfBins
        maxNum = i + n if not overflow else lengthOfBins
        maxNum2 = 0 if not overflow else (i+n) % lengthOfBins
        binRange = np.concatenate((np.arange(i, maxNum), np.arange(0, maxNum2)))
        newSum = sum(bins[binRange])

        if newSum > maxSum :
            maxSum = newSum
            maxBins = binRange
        
    return maxBins


# change eye color
def changeEyeColor(img, eye_color, test=False) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)

    centers = []

    eye_radius = 0

    for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            center = ((2*ex + ew)//2, (2*ey + eh)//2)
            eye_radius = ew//4
            centers.append(center)    

    histogram, eye_pixels, eye_pixel_hue, h, s, v = createHistorgramOfEyeColor(roi_color, centers, eye_radius, test)
    return changeHueOfHistogram(histogram, eye_pixels, eye_pixel_hue, h, s, v, eye_color, test)

img = cv2.imread("D:\DownHere\eye_blue.jpg")  
img = cv2.resize(img,None,fx=0.5,fy=0.5)

newImage = changeEyeColor(img, "brown", True)

cv2.imshow('newImg',newImage)
cv2.waitKey(0)



