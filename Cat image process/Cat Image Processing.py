# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 13:57:25 2022

@author: dsouzm3
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def display(windowName, image):
    cv.imshow(windowName,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
 
"""Task 1 - display coloured image"""
image = cv.imread("cat.png")
display("Coloured CAT", image)

"""display gray scaled image """
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
display('Grayscaled CAT', gray_image)

"""blurring using positive kernel"""
kernel = np.array(np.ones((5,5)))/80
blurred_image = cv.filter2D(gray_image, -1, kernel)
display('Blurred CAT', blurred_image)

"""Task 2 - sobel fileter for vertical and horizontal edges"""
vertical_filter = np.array([[1,0,-1],
                            [2,0,-2],
                            [1,0,-1]])
vertical_image = cv.filter2D(gray_image, -1, vertical_filter)
display("Vertical Edges", vertical_image)

horizontal_filter = np.array([[1,2,1],
                            [0,0,0],
                            [-1,-2,-1]])
horizontal_image = cv.filter2D(gray_image, -1, horizontal_filter)
display("Horizontal Edges", horizontal_image)

"""Task 3 - Create gaussian derivative filters"""
#Guassian smoothening filter
sigma = 1
x,y = np.meshgrid(np.arange(0,len(gray_image[0])), np.arange(0,len(gray_image)))
smooth_kernel = np.exp(-((x-len(gray_image[0])/2)**2 + (y-len(gray_image)/2)**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))
smooth_image = cv.filter2D(gray_image, -1, smooth_kernel)
plt.imshow(smooth_kernel)
plt.show()
display("Smoothened CAT", smooth_image)

#deriveatives Guassian edge detecting filter
sigma = 1
# x,y = np.meshgrid(np.arange(0,30), np.arange(0,30))
x,y = np.meshgrid(np.arange(0,len(gray_image[0])), np.arange(0,len(gray_image)))
x_derv_gauss = (np.exp(-((x-len(gray_image[0])/2)**2 + (y-len(gray_image)/2)**2)/(2*(sigma**2))) * (-(x-len(gray_image[0])/2))) / (2*np.pi*(sigma**4))
x_derv_image = cv.filter2D(gray_image, -1, x_derv_gauss)
plt.imshow(x_derv_gauss)
plt.show()
display("X_dervivative CAT", x_derv_image)

# x,y = np.meshgrid(np.arange(0,len(gray_image[0])), np.arange(0,len(gray_image)))
y_derv_gauss = (np.exp(-((x-len(gray_image[0])/2)**2 + (y-len(gray_image)/2)**2)/(2*(sigma**2))) * (-(y-len(gray_image)/2))) / (2*np.pi*(sigma**4))
y_derv_image = cv.filter2D(gray_image, -1, y_derv_gauss)
plt.imshow(y_derv_gauss)
plt.show()
display("Y_drivative CAT", y_derv_image)

#Direction Independent edge detection using gaussian derivatives
dir_independent_edges = (x_derv_image**2) + (y_derv_image**2)
display("Direction Independent CAT", dir_independent_edges)

# """Fourier transform and Inverse"""
# ft_img = np.fft.fftshift(np.fft.fft2(gray_image))
# # fshift = np.fft.fftshift(ft_img)
# # magnitude_spectrum = 20*np.log(np.abs(fshift))
# result = abs(20*np.log(ft_img))/255
# plt.imshow(ft_img, cmap='Dark2')
# plt.show()  




