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

def main():
    """Task 1 - display coloured image"""
    rgb_image = cv.imread("cat.png")
    display("Coloured CAT", rgb_image)
    
    """display grey scaled image """
    grey_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY)
    display('Greyscaled CAT', grey_image)

    positive_blur(rgb_image, 50, "RBG")
    positive_blur(grey_image, 20, "GREY")
    
    sobel_filter(rgb_image, "RGB")
    sobel_filter(grey_image, "Grey")
    
    gauss_blur(rgb_image, 2, 7, "RGB")
    gauss_blur(grey_image, 4, 5, "Grey")
    
    rgb_x = x_derivatives_gauss(rgb_image, 1, 7, "RGB")
    grey_x = x_derivatives_gauss(grey_image, 1, 7, "Grey")
    
    rgb_y = y_derivatives_gauss(rgb_image, 1, 7, "RGB")
    grey_y = y_derivatives_gauss(grey_image, 1, 7, "Grey")
    
    direction_independent_gauss(rgb_x, rgb_y, "RGB")
    direction_independent_gauss(grey_x, grey_y, "Grey")
    
def display(windowName, image):
    cv.imshow(windowName, image)
    savename  = windowName+'.jpg'
    print(savename)
    cv.waitKey(0)
    cv.imwrite(savename, image)
    cv.destroyAllWindows()
 
def positive_blur(image, den, window):
    """blurring using positive kernel"""
    name  = window +' Blurred CAT'
    kernel = np.array(np.ones((5,5)))/den
    blurred_image = cv.filter2D(image, -1, kernel)
    display(name, blurred_image)

def sobel_filter(image, window):
    """Task 2 - sobel fileter for vertical and horizontal edges"""
    name = window + " Vertical Edges"
    vertical_filter = np.array([[1,0,-1],
                                [2,0,-2],
                                [1,0,-1]])
    vertical_image = cv.filter2D(image, -1, vertical_filter)
    display (name, vertical_image)
    
    name = window + " Horizontal Edges"
    horizontal_filter = np.array([[1,2,1],
                                  [0,0,0],
                                  [-1,-2,-1]])
    horizontal_image = cv.filter2D(image, -1, horizontal_filter)
    display(name, horizontal_image)

def gauss_blur(image, sigma, filter_size, window):
    """Task 3 - Create gaussian derivative filters"""
    name =  window + " Gauss Smoothened CAT"
    x,y = np.meshgrid(np.arange(0,filter_size), np.arange(0,filter_size))
    smooth_kernel = np.exp(-((x-np.mean(x))**2 + (y-np.mean(y))**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))
    smooth_image = cv.filter2D(image, -1, smooth_kernel)
    plt.imshow(smooth_kernel)
    plt.title("Discrete approximation to Gaussian function")
    plt.show()
    display(name, smooth_image)

def x_derivatives_gauss(image, sigma, filter_size, window):    
    #deriveatives Guassian edge detecting filter
    name = window+ " X_dervivative CAT"
    x,y = np.meshgrid(np.arange(0,filter_size), np.arange(0,filter_size))
    x_derv_gauss = (np.exp(-((x-np.mean(x))**2 + (y-np.mean(y))**2)/(2*(sigma**2))) * (-(x-np.mean(x)))) / (2*np.pi*(sigma**4))
    x_derv_image = cv.filter2D(image, -1, x_derv_gauss)
    plt.imshow(x_derv_gauss)
    plt.title("X_dervivative Kernel")
    plt.show()
    display(name, x_derv_image)
    return x_derv_image

def y_derivatives_gauss(image, sigma, filter_size, window):    
    #deriveatives Guassian edge detecting filter
    name = window+ " Y_dervivative CAT"
    x,y = np.meshgrid(np.arange(0,filter_size), np.arange(0,filter_size))
    y_derv_gauss = (np.exp(-((x-np.mean(x))**2 + (y-np.mean(y))**2)/(2*(sigma**2))) * (-(y-np.mean(y)))) / (2*np.pi*(sigma**4))
    y_derv_image = cv.filter2D(image, -1, y_derv_gauss)
    plt.imshow(y_derv_gauss)
    plt.title("Y_dervivative Kernel")
    plt.show()
    display(name, y_derv_image)
    return y_derv_image

def direction_independent_gauss(x_derivative, y_derivative, window):
    name = window + " Direction Independent CAT"
    #Direction Independent edge detection using gaussian derivatives
    dir_independent_edges = (x_derivative**2) + (y_derivative**2)
    display(name, dir_independent_edges)

# """Fourier transform and Inverse"""
# ft_img = np.fft.fftshift(np.fft.fft2(gray_image))
# # fshift = np.fft.fftshift(ft_img)
# # magnitude_spectrum = 20*np.log(np.abs(fshift))
# result = abs(20*np.log(ft_img))/255
# plt.imshow(ft_img, cmap='Dark2')
# plt.show()  

if __name__ == '__main__':
    main()
