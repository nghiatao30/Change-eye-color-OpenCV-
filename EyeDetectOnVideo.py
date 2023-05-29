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
        # if test:
        #     cv2.imshow('test',gray)
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
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  
    centers = []
    eye_radius = 0

    eyes = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        center = ((2*ex + ew)//2, (2*ey + eh)//2)
        eye_radius = int(ew//3.8)
        centers.append(center)
            
    histogram, eye_pixels, eye_pixel_hue, h, s, v = createHistorgramOfEyeColor(img, centers, eye_radius, test)
    return changeHueOfHistogram(histogram, eye_pixels, eye_pixel_hue, h, s, v, eye_color, test)



video_cap = cv2.VideoCapture("TGMT\cute Blue eyes Girl Sub's for more videos #shorts #girl #hot #viral #buletinawani.mp4")

frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_cap.get(cv2.CAP_PROP_FPS)

# Create a video writer object
output_path = "D:\Py\TGMT\output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width//2, frame_height//2))
while video_cap.isOpened():
        
    success, img = video_cap.read()

    if not success:
        break

    img = cv2.resize(img,None,fx=0.5,fy=0.5)

    newImage = changeEyeColor(img, "brown", True)

    out.write(newImage)     
    
video_cap.release()
out.release()

video_cap2 = cv2.VideoCapture("D:\Py\TGMT\output.mp4")

while video_cap2.isOpened():
    success, img = video_cap2.read()

    if not success:
        break

    cv2.imshow("new",img)
    # Close the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap2.release()
cv2.destroyAllWindows()




