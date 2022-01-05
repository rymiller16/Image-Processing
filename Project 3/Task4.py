#Ryan Miller
#u1067596
#Image Processing 
#Project 3 - Task 4

#Task 4: Mosaic

from skimage.io import imread_collection
from skimage import io
import math
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from skimage.metrics import mean_squared_error

print("You will be asked to select a set of images to mosaic.\n")
print("Set 1 - Cell Images\nSet 2 - Golden Gate Images\n")
pic_input = input("Pick the set to be analyzed: (1 or 2): ")
pic_input = int(pic_input)

if pic_input == 1:
    col_dir = 'Images - Part4/cell_images/*.png'
    cell_image = 1
else:
    col_dir = 'Images - Part4/bridge_images/*.png'
    cell_image = 0

#creating a collection with the available images
col = imread_collection(col_dir)
image_matrix = list(range(len(col)))

max_w = 0
max_h = 0
for i in range(0,len(col)):
    image_matrix[i] = col[i]
    image_matrix[i] = color.rgb2gray(image_matrix[i])
    H,W = np.shape(image_matrix[i])
    max_w = max_w + W
    max_h = max_h + H

canvas = np.zeros((max_w,max_h))

# def phase_correlation(a, b):
#     F = np.fft.fft2(a)
#     G = np.fft.fft2(b)
#     conj_F = np.conjugate(F)
#     R = conj_F*G
#     R /= np.absolute(R)
    
#     #lowpass filter
#     H,W = np.shape(a)
#     distance = H
#     window1d_kaiser = np.kaiser(distance,100)
#     window2d_kaiser = np.sqrt(np.outer(window1d_kaiser,window1d_kaiser))

#     R = R*window2d_kaiser
#     r = np.fft.ifft2(R).real
#     return r

def phase_correlation(a, b):
    F = np.fft.fft2(a)
    G = np.fft.fft2(b)
    conj_F = np.conjugate(F)
    R = conj_F*G
    R /= np.absolute(R)
    
    #lowpass filter
    H,W = np.shape(a)
    distance = H
    window1d_kaiser = np.kaiser(distance,100)
    window2d_kaiser = np.sqrt(np.outer(window1d_kaiser,window1d_kaiser))
    
    filter_matrix1 = np.zeros((H,W))
    filter_matrix1 = filter_matrix1+0.01
    radius = 25

    for x in range(int(W/2)-radius,int(W/2)+radius):
        for y in range(int(H/2)-radius,int(H/2)+radius):
            x_ = x - int(W/2)
            y_ = y - int(H/2)
            square = x_**2+y_**2
            dist = math.sqrt(square)
            if dist <= radius:
                filter_matrix1[y,x] = 1

    R = R*window2d_kaiser
    #R = R*filter_matrix1
    r = np.fft.ifft2(R).real

    canvas = np.zeros((2*H,2*W))
    for x in range(0,W):
        for y in range(0,H):
            canvas[x,y] = a[x,y]
    for x in range(W,2*W):
        for y in range(H,2*H):
            canvas[x,y] = a[x-W,y-H]
    for x in range(W,2*W):
        for y in range(0,H):
            canvas[x,y] = a[x-W,y]
    for x in range(0,W):
        for y in range(H,2*H):
            canvas[x,y] = a[x,y-H]
    
    x, y = (np.where(r==np.amax(r)))        #x,y of phase correlation peak
    x = int(x)
    y = int(y)
    trans_x = W - x
    trans_y = H - y
    
    canvas2 = np.ones((2*H,2*W))
    for x in range(0+trans_x,W+trans_x):
        for y in range(0+trans_y,H+trans_y):
            canvas2[x,y] = b[x-trans_x,y-trans_y]
    
    x, y = (np.where(r==np.amax(r)))
    x = int(x)
    y = int(y)

    list_1 = list(range(0,trans_x)) + list(range(trans_x+W,2*W))
    list_2 = list(range(0,trans_y)) + list(range(trans_y+H,2*H))
    canvas = np.delete(canvas,(list_1), axis = 0)
    canvas = np.delete(canvas,(list_2), axis = 1)
    canvas2 = np.delete(canvas2,(list_1), axis = 0)
    canvas2 = np.delete(canvas2,(list_2), axis = 1)
    
    MSE_domains = np.zeros(4)
    corr_domains = np.zeros(4)
    counter = 0 # 0 - Top left, 1 - Top right, 2 - bottom left, 3 - bottom right
    
    #first subdomain (Top Left)
    list_s1 = list(range(x,W))
    list_s2 = list(range(y,H))
    canvas_s1 = np.delete(canvas,(list_s1),axis=0)
    canvas2_s1 = np.delete(canvas2,(list_s2),axis=1)
    canvas_s1 = np.delete(canvas_s1,(list_s2),axis=1)
    canvas2_s1 = np.delete(canvas2_s1,(list_s1),axis=0)
    MSE_domains[counter] = mean_squared_error(canvas_s1, canvas2_s1)
    # H_1,W_1 = np.shape(canvas_s1)
    # H_2,W_2 = np.shape(canvas2_s1)
    # print(H_1)
    # print(W_1)
    # print(H_2)
    # print(W_2)
    corr_domains[counter] = norm_correlation(canvas_s1, canvas2_s1)
    counter = counter+1
    
    #second subdomain (Top Right)
    list_s1 = list(range(0,x))
    list_s2 = list(range(y,H))
    canvas_s1 = np.delete(canvas,(list_s1),axis=0)
    canvas2_s1 = np.delete(canvas2,(list_s2),axis=1)
    canvas_s1 = np.delete(canvas_s1,(list_s2),axis=1)
    canvas2_s1 = np.delete(canvas2_s1,(list_s1),axis=0)
    MSE_domains[counter] = mean_squared_error(canvas_s1, canvas2_s1)
    corr_domains[counter] = norm_correlation(canvas_s1, canvas2_s1)
    counter = counter+1
    
    #third subdomain (Bottom Left)
    list_s1 = list(range(x,W))
    list_s2 = list(range(0,y))
    canvas_s1 = np.delete(canvas,(list_s1),axis=0)
    canvas2_s1 = np.delete(canvas2,(list_s2),axis=1)
    canvas_s1 = np.delete(canvas_s1,(list_s2),axis=1)
    canvas2_s1 = np.delete(canvas2_s1,(list_s1),axis=0)
    MSE_domains[counter] = mean_squared_error(canvas_s1, canvas2_s1)
    corr_domains[counter] = norm_correlation(canvas_s1, canvas2_s1)
    counter = counter+1
    
    #fourth subdomain (Bottom Right)
    list_s1 = list(range(0,x))
    list_s2 = list(range(0,y))
    canvas_s1 = np.delete(canvas,(list_s1),axis=0)
    canvas2_s1 = np.delete(canvas2,(list_s2),axis=1)
    canvas_s1 = np.delete(canvas_s1,(list_s2),axis=1)
    canvas2_s1 = np.delete(canvas2_s1,(list_s1),axis=0)
    MSE_domains[counter] = mean_squared_error(canvas_s1, canvas2_s1)
    corr_domains[counter] = norm_correlation(canvas_s1, canvas2_s1)
    counter = counter+1
    
    # C1 = np.fft.fft2(canvas)
    # C2 = np.fft.fft2(canvas2)
    # R2 = C1*C2
    # r2 = np.fft.ifft2(R2).real
    
    max_MSE = np.argmin(MSE_domains, axis=0)
    if max_MSE == 0:    #TopL
        loc = "TopL"
    elif max_MSE == 1:  #TopR
        loc = "TopR"
    elif max_MSE == 2:  #BotL
        loc = "BotL"
    elif max_MSE == 3:  #BotR
        loc = "BotR"
        
    x, y = (np.where(r==np.amax(r)))        #x,y of phase correlation peak
    x = int(x)
    y = int(y)
    
    return r, loc, x, y, corr_domains

def norm_correlation(a,b):
    num = 0
    mean_a = np.sum(a)/a.size
    mean_b = np.sum(b)/b.size
    a_hat = a - mean_a
    b_hat = b - mean_b
    H_1,W_1 = np.shape(a_hat)
    H_2,W_2 = np.shape(b_hat)
    a_inside = a_hat**2
    b_inside = b_hat**2
    sum_a = np.sum(a_inside)
    sum_b = np.sum(b_inside)
    sqrt_a = math.sqrt(sum_a)
    sqrt_b = math.sqrt(sum_b)
    a_bar = sqrt_a
    b_bar = sqrt_b
        
    for x in range(0,W_1-1):
        for y in range(0,H_1-1):
            num = num + (a_hat[y,x]*b_hat[y,x])/(a_bar*b_bar)
    # num = np.matmul(a_hat,b_hat)
    # den = sqrt_a*sqrt_b
    # result = np.divide(num,den)
    # final = np.sum(result)
    return num

def MSE_attempt(a,b):
    mse_val = mean_squared_error(a,b)
    return mse_val

#choose anchor, eliminate from the active set, and put into canvas
active_set = np.zeros(len(col))
# anchor = randint(0, len(col)-1)
anchor = 1
active_set[anchor] = -1
anchor_image = image_matrix[anchor]
H_can, W_can = np.shape(canvas)
H_mid = int(H_can/2)
W_mid = int(W_can/2)

x_trans = W_mid-int(W/2)
y_trans = H_mid-int(H/2)

for x in range(W_mid-int(W/2),W_mid+int(W/2)):
    for y in range(H_mid-int(H/2),H_mid+int(H/2)):
        stuff_x = (x-x_trans)
        stuff_y = (y-y_trans)
        canvas[x,y] = anchor_image[x-x_trans,y-y_trans]
    

corr_array = np.zeros(len(col))
counter = 0
counter2 = 0
counter_negs = 0
max_corr = 0
ref2_ = 0

while(1):
    for i in range(0,len(col)):
        if active_set[i] != -1: #if active
            #corr_array[i] = norm_correlation(anchor_image,image_matrix[i])
            corr_array[i] = MSE_attempt(anchor_image,image_matrix[i])
            if counter == 0:
                r, loc, x, y, corr_domains = phase_correlation(anchor_image, image_matrix[i])
                ref_ = corr_array[i]
                ref_idx = i
                counter = counter + 1
                max_corr = np.max(corr_domains)
            else: 
                #if corr_array[i] > ref_:
                #if corr_array[i] < ref_:
                r, loc, x, y, corr_domains = phase_correlation(anchor_image, image_matrix[i])
                for j in range(0,3):
                    if corr_domains[j] > max_corr:   #yes there is overlap (might need to change this threshold)
                        if counter2 == 0:
                            ref2_ = corr_domains[j]
                            ref_ = corr_array[i]
                            ref_idx = i
                            counter2 = counter2 + 1
                        else:
                            if corr_domains[j] > ref2_:
                                ref2_ = corr_domains[j]
                                ref_ = corr_array[i]
                                ref_idx = i
                    
        else: 
            corr_array[i] = -1  
            counter_negs = counter_negs + 1
            if counter_negs == len(col):
                break
    
    if counter_negs == len(col):
        break
    r, loc, x, y, corr_domains = phase_correlation(anchor_image, image_matrix[ref_idx])
    
    if cell_image == 1 and counter_negs == len(col)-1:
        print("Hello")
        x = 399
        y = 0
        x_trans = 1275
        y_trans = 1275
        loc = "BotR"
    
    anchor_image = image_matrix[ref_idx]
    
    if loc == "TopL":   
        x_trans = x_trans + (W-x)
        y_trans = y_trans + (H-y)
        for x in range(x_trans,x_trans+W):
            for y in range(y_trans,y_trans+H):
                canvas[x,y] = anchor_image[x-x_trans,y-y_trans]     
    
    elif loc == "TopR": 
        x_trans = x_trans - x
        y_trans = y_trans + (H-y)
        for x in range(x_trans,x_trans+W):
            for y in range(y_trans,y_trans+H):
                canvas[x,y] = anchor_image[x-x_trans,y-y_trans]
    
    elif loc == "BotL":
        x_trans = x_trans + (W-x)
        y_trans = y_trans - y
        for x in range(x_trans,x_trans+W):
            for y in range(y_trans,y_trans+H):
                canvas[x,y] = anchor_image[x-x_trans,y-y_trans]
    
    elif loc == "BotR":
        x_trans = x_trans - x  #+->-
        y_trans = y_trans - y
        for x in range(x_trans,x_trans+W):
            for y in range(y_trans,y_trans+H):
                x_ = x-x_trans
                y_ = y-y_trans
                canvas[x,y] = anchor_image[x-x_trans,y-y_trans]
    
    counter = 0
    counter2 = 0
    active_set[ref_idx] = -1
    counter_negs = 0
    max_corr = 0
    ref2_ = 0


if cell_image == 0:
    H,W = np.shape(canvas)
    list_1 = list(range(0,1150)) + list(range(1800,W))
    list_2 = list(range(0,1050)) + list(range(1800,H))
    canvas = np.delete(canvas,(list_1), axis = 0)
    canvas = np.delete(canvas,(list_2), axis = 1)
if cell_image == 1:
    H,W = np.shape(canvas)
    list_1 = list(range(0,850)) + list(range(2200,W))
    list_2 = list(range(0,1200)) + list(range(2200,H))
    canvas = np.delete(canvas,(list_1), axis = 0)
    canvas = np.delete(canvas,(list_2), axis = 1)
plt.imshow(canvas,cmap='gray')





            



    
    
                
    