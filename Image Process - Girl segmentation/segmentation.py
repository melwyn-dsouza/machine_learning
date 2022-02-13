# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 13:00:08 2022

@author: dsouzm3
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

cwd = os.getcwd()
savepath =cwd+"\\results\\"
print(savepath)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def main():
    """Task 1 - display input and target image, get hsv channel and display"""
    inputImage = cv.imread("Girl_in_front_of_a_green_background.jpg")
    targetImage = cv.imread("Tour_Eiffel.jpg")
    display("Girl", inputImage)
    display("Eiffel", targetImage)
    
    hsvImage = hsv(inputImage, "HSV of input image")
    h,s,v = cv.split(hsvImage)
    display("Hue channel", h)
    display("Saturation channel", s)
    display("V channel", v)
    
    """Task 2 - find mask to remove green background"""
    histImage = hist(h, "Hue Image hist")
    
    threshold1, threshold2 = 50, 80
    
    plt.bar(range(len(histImage)),histImage.flatten(), label ="Histogram")
    mask = np.ones((len(histImage),1))
    mask[threshold1:threshold2] = (max(histImage[threshold1:threshold2]))
    plt.bar(range(len(histImage)), mask.flatten(), color = 'green',  alpha=0.5, label = "Mask")
    plt.title("Hue Image hist with green mask")
    plt.legend()
    plt.show()
    
    ret1, img1 = cv.threshold(h, threshold1, 255, cv.THRESH_BINARY_INV)
    display('Theshold '+str(ret1), img1)
    hist(img1,"img1")
    ret2, img2 = cv.threshold(h, threshold2, 255, cv.THRESH_BINARY)
    display('Theshold '+str(ret2), img2)
    hist(img2,"img2")
    foreground = img1+img2
    display('Foreground Mask using thresholds', foreground)
    hist(foreground, "Foreground Hist")
    
    foreground = foreground.reshape(960, 640, 1)
    mask = foreground > 0
    maskedImage = inputImage*mask
    display("Foreground of Input Image", maskedImage)
    
    """Task 3 - Final Image"""
    resizedImage = cv.resize(maskedImage, (200,300))
    display("Resized Foreground", resizedImage)
    finalImage = copy.deepcopy(targetImage)
    
    for i in range(233,533):
        for j in range(60,260):
            if resizedImage[i-233][j-60][1] > 0:
                finalImage[i][j] = resizedImage[i-233][j-60][:]
    display("Final Image", finalImage)
    
def hsv(image, windowname):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return hsv_image

def hist(image, title):
    hist = cv.calcHist([image],[0],None,[256],[0,256])
    plt.bar(range(len(hist)),hist.flatten(), label ="Histogram"); plt.title(title); plt.show()
    return hist
    
def display(windowName, image, save = True):
    global savepath
    cv.imshow(windowName, image)
    cv.waitKey(0)
    if save:
        savename  = savepath+windowName+'.png'
        if np.max(image) == 1:
            image = image*255
        cv.imwrite(savename, image)
    cv.destroyAllWindows()
    
if __name__ == '__main__':
    main()
