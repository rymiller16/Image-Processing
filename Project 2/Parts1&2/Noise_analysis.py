#Ryan Miller
#Analyzing Noise

import skimage 
from skimage import io
from skimage import color
from skimage import measure
import matplotlib.pyplot as plt 

def MSE(noisy_image,ground_image):
    MSE = skimage.metrics.mean_squared_error(ground_image,noisy_image)
    return MSE

def SS(noisy_image,ground_image):
    SS = skimage.metrics.structural_similarity(ground_image,noisy_image)
    return SS

print("You will be asked to pick a picture to be analyzed and type of noise to be applied.\n")
print("Picture 1 - Jerry Garcia\nPicture 2 - Jim Morrison\nPicture 3 - Mark Knopfler\nPicture 4 - Neil Young\nPicture 5 - Mick Jagger\n")
pic_input = input("Pick a picture to be analyzed: (1-5): ")
pic_input = int(pic_input)

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
    
mean = 0.25
var = 0.5
amount = 0.25
if noise_type == 1:     #Gauss
    noisy_image = skimage.util.random_noise(ref_image,mode='gaussian', mean = mean, var = var)
elif noise_type == 2:   #Salt and Pepper
    noisy_image = skimage.util.random_noise(ref_image,mode='s&p', amount = amount)
elif noise_type == 3:   #Poisson
    noisy_image = skimage.util.random_noise(ref_image,mode='poisson')
elif noise_type == 4:   #Speckle
    noisy_image = skimage.util.random_noise(ref_image,mode='speckle', mean = mean, var = var)

fig = plt.figure(figsize = (12,12))
rows = 1
columns = 2

fig.add_subplot(rows,columns,1)
plt.imshow(ref_image,cmap ="gray")
MSE_ref = MSE(ref_image,ref_image)
SS_ref = SS(ref_image,ref_image)
plt.title('Original Image')
plt.xlabel('MSE: %1.3f, SS: %1.3f' %(MSE_ref,SS_ref), fontsize = 12)
fig.tight_layout(pad=3.0)

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