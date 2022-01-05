#Ryan Miller
#Image Processing 

#This is a script with the intent purpose of applying linear filters to images

#Libraries
import skimage 
from skimage import io
from skimage import color
from skimage import measure
import matplotlib.pyplot as plt 
from skimage.segmentation import flood_fill
import numpy as np
from skimage.measure import label, regionprops
from skimage import morphology
from skimage.util import random_noise
from skimage import color,data,restoration
from skimage.filters import median 
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle, denoise_tv_bregman, denoise_wavelet

def MSE(noisy_image,ground_image):
    MSE = skimage.metrics.mean_squared_error(ground_image,noisy_image)
    return MSE

def SS(noisy_image,ground_image):
    SS = skimage.metrics.structural_similarity(ground_image,noisy_image)
    return SS

#Take input from User -> Picture and which type of filtering
print("You will be asked to pick a picture to be analyzed and type of filtering to be applied.\n")
print("Picture 1 - Jerry Garcia\nPicture 2 - Jim Morrison\nPicture 3 - Mark Knopfler\nPicture 4 - Neil Young\nPicture 5 - Mick Jagger\n")
pic_input = input("Pick a picture to be analyzed: (1-5): ")
pic_input = int(pic_input)

filter_input = input("\nWhat type of filters would you like to apply (linear(enter 1) or non-linear(enter 2)): ")
filter_input = int(filter_input)

if filter_input == 1:  #Linear
    noise_type = input("\nChoose noise type (1-Gaussian, 2-Salt & Pepper, 3-Poisson, 4-Speckle): ")
    noise_type = int(noise_type)
elif filter_input == 2: #Non-linear
    noise_type = input("\nChoose noise type (1-Gaussian, 2-Salt & Pepper, 3-Poisson, 4-Speckle): ")
    noise_type = int(noise_type)

#Read in the image to be analyzed
if pic_input == 1:
    img = io.imread("Jerry.jpg")
    ref_image = color.rgb2gray(img)
elif pic_input == 2:
    img = io.imread("Jim.jpg")
    ref_image = color.rgb2gray(img)
elif pic_input == 3:
    img = io.imread("Mark.jpg")
    ref_image = color.rgb2gray(img)
elif pic_input == 4:
    img = io.imread("Neil.jpg")
    ref_image = color.rgb2gray(img)
elif pic_input == 5:
    img = io.imread("Mick.jpg")
    ref_image = color.rgb2gray(img)

if filter_input == 1:   #linear 
    
# 3 options, amount = 0.1, 0.25, 0.5 - salt and pepper
# 3 options, mean = 0.1, var = 0.5/ mean 0.25, var = 0.25/ mean = 0.5, var = 0.1 - gaussian and speckle
    mean = 0.5
    var = 0.1
    amount = 0.5
    
    if noise_type == 1:     #Gauss
        noisy_image = skimage.util.random_noise(ref_image,mode='gaussian', mean = mean, var = var)
    elif noise_type == 2:   #Salt and Pepper
        noisy_image = skimage.util.random_noise(ref_image,mode='s&p', amount = amount)
    elif noise_type == 3:   #Poisson
        noisy_image = skimage.util.random_noise(ref_image,mode='poisson')
    elif noise_type == 4:   #Speckle
        noisy_image = skimage.util.random_noise(ref_image,mode='speckle', mean = mean, var = var)
    
    #plotting with multiple figures
    fig = plt.figure(figsize = (12,12))
    rows = 3
    columns = 3
    
    #1,1 position - Original Image
    fig.add_subplot(rows,columns,1)
    plt.imshow(ref_image,cmap ="gray")
    MSE_ref = MSE(ref_image,ref_image)
    SS_ref = SS(ref_image,ref_image)
    plt.title('Original Image')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_ref,SS_ref), fontsize = 12)
    fig.tight_layout(pad=3.0)
    
    #1,2 position - Noisy Image
    fig.add_subplot(rows,columns,2)
    plt.imshow(noisy_image,cmap="gray")
    MSE_noisy = MSE(noisy_image,ref_image)
    SS_noisy = SS(noisy_image,ref_image)
    if noise_type == 1:     #Gauss
        plt.title('Gaussian Noise - Mean: %1.2f, Var: %1.2f' %(mean, var))
    elif noise_type == 2:   #Salt and Pepper
        plt.title('Salt and Pepper Noise - Amount: %1.2f' %amount)
    elif noise_type == 3:   #Poisson
        plt.title('Poisson Noise')
    elif noise_type == 4:   #Speckle
        plt.title('Speckle Noise - Mean: %1.2f, Var: %1.2f' %(mean, var))
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_noisy,SS_noisy), fontsize = 12)
    
    #1,3 - Box filter - correlate (3x3)
    kernel = np.ones((11,11), np.float32) /121      #box filter 3x3, values = 1/9
    box_image = skimage.filters.correlate_sparse(noisy_image,kernel)
    fig.add_subplot(rows,columns,3)
    plt.imshow(box_image,cmap="gray")
    MSE_box_image = MSE(box_image,ref_image)
    SS_box_image = SS(box_image,ref_image)
    plt.title('11x11 box filter [1/121]')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_box_image,SS_box_image), fontsize = 12)
    
    #2,1 - Derivative filter (3x3)
    kernel = [[0,0,0],[-1,0,1],[0,0,0]]      #box filter 3x3, values = 1/9
    box_image = skimage.filters.correlate_sparse(noisy_image,kernel)
    fig.add_subplot(rows,columns,4)
    plt.imshow(box_image,cmap="gray")
    MSE_box_image = MSE(box_image,ref_image)
    SS_box_image = SS(box_image,ref_image)
    plt.title('3x3 Derivative filter')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_box_image,SS_box_image), fontsize = 12)
    
    
    #2,2 - Wiener filter
    psf = np.ones((5,5))/25
    deconvolved_img = restoration.wiener(noisy_image,psf,50)
    fig.add_subplot(rows,columns,5)
    plt.imshow(deconvolved_img,cmap='gray') 
    MSE_wiener = MSE(deconvolved_img,ref_image)
    SS_wiener = SS(deconvolved_img, ref_image)
    plt.title('Wierner Filter')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_wiener,SS_wiener), fontsize = 12)
    
    #2,3 - Gaussian filter
    gaussian_image = skimage.filters.gaussian(noisy_image, sigma = 1)
    fig.add_subplot(rows,columns,6)
    plt.imshow(gaussian_image,cmap="gray")
    MSE_gaussian = MSE(gaussian_image,ref_image)
    SS_gaussian = SS(gaussian_image,ref_image)
    plt.title('Gaussian Filter: Sigma = 1')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_gaussian,SS_gaussian), fontsize = 12)
    
    #3,1 - Gaussian filter
    gaussian_image = skimage.filters.gaussian(noisy_image, sigma = 0.5)
    fig.add_subplot(rows,columns,7)
    plt.imshow(gaussian_image,cmap="gray")
    MSE_gaussian = MSE(gaussian_image,ref_image)
    SS_gaussian = SS(gaussian_image,ref_image)
    plt.title('Gaussian Filter: Sigma = 0.5')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_gaussian,SS_gaussian), fontsize = 12)
    
    #3,2 - Rectangular Filter
    kernel = np.ones((5,1))/5      #rectangular filter 5x3 = 15
    rect_image = skimage.filters.correlate_sparse(noisy_image,kernel)
    fig.add_subplot(rows,columns,8)
    plt.imshow(box_image,cmap="gray")
    MSE_rect_image = MSE(rect_image,ref_image)
    SS_rect_image = SS(rect_image,ref_image)
    plt.title('5x3 Rectangular filter [1/15]')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_rect_image,SS_rect_image), fontsize = 12)
    
    #3,3 combination of derivatives 
    matrix1 = [[0,0,0],[-1,0,1],[0,0,0]]
    matrix2 = [[0,-1,0],[0,0,0],[0,1,0]]
    kernel = np.dot(matrix1,matrix2)
    mult_image = skimage.filters.correlate_sparse(noisy_image,kernel)
    fig.add_subplot(rows,columns,9)
    plt.imshow(mult_image,cmap="gray")
    MSE_mult_image = MSE(mult_image,ref_image)
    SS_mult_image = SS(mult_image,ref_image)
    plt.title('Partial Derivatives Filter')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_mult_image,SS_mult_image), fontsize = 12)


elif filter_input == 2: #non-linear

# 3 options, amount = 0.1, 0.25, 0.5 - salt and pepper
# 3 options, mean = 0.1, var = 0.5/ mean 0.25, var = 0.25/ mean = 0.5, var = 0.1 - gaussian and speckle

    mean = 0.5
    var = 0.1
    amount = 0.5
    
    if noise_type == 1:     #Gauss
        noisy_image = skimage.util.random_noise(ref_image,mode='gaussian', mean = mean, var = var)
    elif noise_type == 2:   #Salt and Pepper
        noisy_image = skimage.util.random_noise(ref_image,mode='s&p', amount = amount)
    elif noise_type == 3:   #Poisson
        noisy_image = skimage.util.random_noise(ref_image,mode='poisson')
    elif noise_type == 4:   #Speckle
        noisy_image = skimage.util.random_noise(ref_image,mode='speckle', mean = mean, var = var)
    
    #plotting with multiple figures
    fig = plt.figure(figsize = (12,12))
    rows = 3
    columns = 3
    
    #1,1 position - Original Image
    fig.add_subplot(rows,columns,1)
    plt.imshow(ref_image,cmap ="gray")
    MSE_ref = MSE(ref_image,ref_image)
    SS_ref = SS(ref_image,ref_image)
    plt.title('Original Image')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_ref,SS_ref), fontsize = 12)
    fig.tight_layout(pad=3.0)
    
    #1,2 position - Noisy image
    fig.add_subplot(rows,columns,2)
    plt.imshow(noisy_image,cmap="gray")
    MSE_noisy = MSE(noisy_image,ref_image)
    SS_noisy = SS(noisy_image,ref_image)
    if noise_type == 1:     #Gauss
        plt.title('Gaussian Noise - Mean: %1.2f, Var: %1.2f' %(mean, var))
    elif noise_type == 2:   #Salt and Pepper
        plt.title('Salt and Pepper Noise - Amount: %1.2f' %amount)
    elif noise_type == 3:   #Poisson
        plt.title('Poisson Noise')
    elif noise_type == 4:   #Speckle
        plt.title('Speckle Noise - Mean: %1.2f, Var: %1.2f' %(mean, var))
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_noisy,SS_noisy), fontsize = 12)

    #1,3 - Median Filter
    median_image = median(noisy_image,disk(7))
    fig.add_subplot(rows,columns,3)
    plt.imshow(median_image,cmap="gray")
    MSE_median = MSE(median_image,ref_image)
    SS_median = SS(median_image,ref_image)
    plt.title('Median Filter')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_median, SS_median), fontsize = 12)
    
    #2,1 - Bilateral 
    bilateral_image = skimage.restoration.denoise_bilateral(noisy_image)
    fig.add_subplot(rows,columns,4)
    plt.imshow(bilateral_image,cmap="gray")
    MSE_bilateral = MSE(bilateral_image,ref_image)
    SS_bilateral = SS(bilateral_image,ref_image)
    plt.title('Bilateral Filter')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_bilateral, SS_bilateral), fontsize = 12)
    
    #2,2 and 2,3 - non local 
    #we take the noisy image and we try and estimate the stnd deviation of the noise 
    sigma_est = np.mean(estimate_sigma(noisy_image))
    patch_kw = dict(patch_size = 5,patch_distance = 6)   #5x5 patches, 13x13 search area
    
    #slow algorithim 
    denoise_slow = denoise_nl_means(noisy_image, h=0.8*sigma_est, sigma = sigma_est, fast_mode = False, **patch_kw)
    fig.add_subplot(rows,columns,5)
    plt.imshow(denoise_slow,cmap="gray")
    MSE_nl_slow = MSE(denoise_slow, ref_image)
    SS_nl_slow = SS(denoise_slow, ref_image)
    plt.title('Non-Local (Slow Algo)')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_nl_slow, SS_nl_slow))
    
    #fast 
    denoise_fast = denoise_nl_means(noisy_image, h=0.8*sigma_est, sigma = sigma_est, fast_mode = True, **patch_kw)
    fig.add_subplot(rows,columns,6)
    plt.imshow(denoise_fast,cmap="gray")
    MSE_nl_fast = MSE(denoise_fast, ref_image)
    SS_nl_fast = SS(denoise_fast, ref_image)
    plt.title('Non-Local (Fast Algo)')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_nl_fast, SS_nl_fast))
    
    #3,1 - Denoise Tv_chambolle 
    denoise_tv_c = denoise_tv_chambolle(noisy_image, weight = 0.25)
    fig.add_subplot(rows,columns,7)
    plt.imshow(denoise_tv_c,cmap="gray")
    MSE_tv_c = MSE(denoise_tv_c, ref_image)
    SS_tv_c = SS(denoise_tv_c, ref_image)
    plt.title('Denoise TV (Chambolle)')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_tv_c, SS_tv_c))
    
    #3,2 - Denoise Tv_bregman
    denoise_tv_b = denoise_tv_bregman(noisy_image, weight = 0.35)
    fig.add_subplot(rows,columns,8)
    plt.imshow(denoise_tv_b,cmap="gray")
    MSE_tv_b = MSE(denoise_tv_b, ref_image)
    SS_tv_b = SS(denoise_tv_b, ref_image)
    plt.title('Denoise TV (Bregman)')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_tv_b, SS_tv_b))
    
    #3,3 - Denoise Wavelet 
    denoise_wave = denoise_wavelet(noisy_image)
    fig.add_subplot(rows,columns,9)
    plt.imshow(denoise_wave,cmap="gray")
    MSE_wave = MSE(denoise_wave, ref_image)
    SS_wave = SS(denoise_wave, ref_image)
    plt.title('Denoise Wavelet')
    plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_wave, SS_wave))
    
