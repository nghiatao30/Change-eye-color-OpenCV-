import cv2,sys
import numpy as np
import math
import matplotlib.pyplot as plt
import imageio
import gif2numpy as gf


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
        # print("ep: ",eyePixels)
        # print("gray: ",gray)
        # get histogram of colors for eye_pixlels
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v  = cv2.split(hsv_image)
        eye_pixel_hue = h[eyePixels]
        # h[0] / h[180] = red, h[60] = green, h[120] = blue
        histogram = np.histogram(eye_pixel_hue, np.arange(180))
        return histogram, eyePixels, eye_pixel_hue, h , s, v 

def changeHueOfHistogram(histogram, eyePixels, eye_pixel_hue, h, s, v, color, test = False) :
    largestBins = findLargestNConsecutiveBins(histogram,30)
    # if test:
        # print(histogram)
        # print(largestBins)
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

    # tear = cv2.imread(r'TGMT\tear.png')
    # tear = cv2.resize(tear, None ,fx=0.5,fy=0.5)
    # print(tear.shape)

    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  
    # faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    centers = []
    eye_radius = 0
    eyes = eye_cascade.detectMultiScale(gray)
    highereye = [99999999,99999999,99999999]
    for (ex,ey,ew,eh) in eyes:
        # cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        center = ((2*ex + ew)//2, (2*ey + eh)//2)
        eye_radius = min(ew,eh)//4
        print(center,eye_radius)
        if center[1] < highereye[1]:
            highereye = center[0],center[1],eye_radius
        centers.append(center)
            
    histogram, eye_pixels, eye_pixel_hue, h, s, v = createHistorgramOfEyeColor(img, centers, eye_radius, test)
    cimg = changeHueOfHistogram(histogram, eye_pixels, eye_pixel_hue, h, s, v, eye_color, test)



    #add gif effect
    # mask = cv2.imread(r'TGMT\blue-eye3-removebg-preview.png', cv2.IMREAD_UNCHANGED)
    # h1,w1,c1 = cimg.shape
    # h3,w3,c3 = mask.shape

    # h=min(h1,h3)
    # w=min(w1,w3)

    #Thay đổi kích thước ảnh theo w,h:
    fg = cv2.resize(cimg,(cimg.shape[1],cimg.shape[0]))
    # mask = cv2.resize(mask,(w,h))
    
    ex,ey,er = highereye
    # print(highereye)

    # ex = int(ex* (w / w1))
    # ey = int(ey * (h / h1))
    # er = int(er *((h/h1) + (w/w1)/2))

    url = ""
    if eye_color == "green":
        url = "file:///D:/DownHere/output-onlinegiftools%20(1).gif"
    elif eye_color == "blue":
        url = "file:///D:/Py/tearblackbgm.gif"

    frames = imageio.mimread(imageio.core.urlopen(url).read(), '.gif')
    # frames,a,b = gf.convert("TGMT/new tear.gif")
    # frames = [cv2.resize(frame,None,fx=0.1,fy=0.1) for frame in frames]
    # ht,wt,channels = frames[0].shape

    fg_h, fg_w, fg_c = fg.shape
    # bg_h, bg_w, bg_c = frames[0].shape
    # top = int(abs(bg_h-fg_h)/2)
    # left = int(abs(bg_w-fg_w)/2)
    # bgs = [frame[top: top + fg_h, left:left + fg_w, 0:3] for frame in frames]
    bgs = [cv2.resize(frame,(fg_w,fg_h))[:,:,0:3] for frame in frames]
    bgs_h,bgs_w,bgs_c = bgs[0].shape
    # print(bgs[0].shape)
    # print(cimg.shape)

    results = []
    tears= []
    # alpha = 0.5

    for i in range(len(bgs)):
        result = fg.copy()
        # result[mask[:,:,3] != 0]   =  result[mask[:,:,3] != 0]
        # bgs[i][mask[:,:,3] == 0] = 0
        # bgs[i][mask[:,:,3] != 0] = 0.5*bgs[i][mask[:,:,3] != 0]
        result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        result[ey + er:,:,:][bgs[i][:bgs_h - (ey + er),:,:] != 0] = bgs[i][:bgs_h - (ey + er),:,:][bgs[i][:bgs_h - (ey + er),:,:] != 0] # decrease y axis of gif effect but not work
        # result[bgs[i]!= 0] = bgs[i][bgs[i] != 0]

        results.append(result)
        tears.append(bgs[i])



    imageio.mimsave('eyeproject.gif', results)
    imageio.mimsave('tearpart.gif', tears)
    return cimg

img = cv2.imread("TGMT/Eyes-Girls-DP-Thumbnail.jpg")  
img = cv2.resize(img,None,fx=0.4,fy=0.4)

newImage = changeEyeColor(img, "green", True)
# img = cv2.resize(img,None,fx=2,fy=2)

cv2.imshow('imag',img)
cv2.imshow('newImg',newImage)
cv2.waitKey(0)



