#Ryan Miller
#u1067596
#Image Processing 
#Dr. Whitaker 
#Project 4 - Morphing

import numpy as np 
import matplotlib.pyplot as plt
from Morphing_Formatting import Morphing_Formatting
from scipy.interpolate import interp2d
from scipy.linalg import svd
from skimage.exposure import equalize_hist

file_name = "morph_params.json"
num_images, img_names, img_list, num_set_corr, num_corrs, all_corrs, output_name, num_outputs = Morphing_Formatting(file_name)

RBF = input("Choose 1 for normal Radial Basis Function and 2 for Gaussian RBF: ")
RBF = int(RBF)
if RBF == 2:
    sigma = input("Enter a desired sigma value for the Gaussian RBF: ")
    sigma = float(sigma)
    

H = len(img_list[0])
W = len(img_list[0][0])

N = num_corrs

#Find average for correspondances
avg_corrs = [[]]*N
img1_corrs = [[]]*N
img2_corrs = [[]]*N
for i in range(N):
    a = float((all_corrs[0][i][0]+all_corrs[1][i][0])/2)
    b = float((all_corrs[0][i][1]+all_corrs[1][i][1])/2)
    avg_corrs[i] = [a,b]
    x_1 = all_corrs[0][i][0]
    y_1 = all_corrs[0][i][1]
    x_2 = all_corrs[1][i][0]
    y_2 = all_corrs[1][i][1]
    img1_corrs[i] = [x_1,y_1]
    img2_corrs[i] = [x_2,y_2]

#Find Forward Transformations from Img_1 to I_t and Img_2 to I_t
def phi_fn(i,j,corrs):
    if RBF == 1:
        eps = 0.0001
        x_i = corrs[i]
        x_i = np.asmatrix(x_i)
        x_j = corrs[j]
        x_j = np.asmatrix(x_j)
        xi_xj = x_i - x_j
        norm = np.linalg.norm(xi_xj)
        norm_log = norm+eps
        output = norm**2 * np.log(norm_log)
    if RBF == 2: 
        eps = 0.0001
        x_i = corrs[i]
        x_i = np.asmatrix(x_i)
        x_j = corrs[j]
        x_j = np.asmatrix(x_j)
        xi_xj = x_i - x_j
        norm = np.linalg.norm(xi_xj)
        norm = norm **2 
        den = 2*(sigma**2)
        output = np.exp(-norm/den)
    return output

def set_B(corrs):
    B = np.zeros((N+3,N+3))
    for i in range(N):
        B[0,i] = corrs[i][0]
        B[1,i] = corrs[i][1]
        B[2,i] = 1
        B[i+3,N] = corrs[i][1]
        B[i+3,N+1] = corrs[i][0]
        B[i+3,N+2] = 1
    for i in range(N,N+3):
        B[0,i] = 0
        B[1,i] = 0
        B[2,i] = 0
    for i in range(0,N):
        for j in range(0,N):
            B[i+3][j] = phi_fn(i,j,corrs)
    return B

#B's for forward B1 - Img 1 to It and B2 - Img 2 to It
B1 = set_B(img1_corrs)
B2 = set_B(img2_corrs)
B_rev = set_B(avg_corrs)

def set_A(B):
    size_A = (N+3)*2
    mid = int(size_A / 2)
    A = np.zeros((size_A,size_A))
    for i in range(N+3):
        for j in range(N+3):
            A[i][j] = B[i][j]
            A[mid+i][mid+j] = B[i][j]
    return A

A1 = set_A(B1)
A2 = set_A(B2)
A_rev = set_A(B_rev)

def set_b(corrs):
    b = np.zeros((2*(N+3),1))
    for i in range(3,N+3):
        b[i] = corrs[i-3][0]
        b[i+N+3] = corrs[i-3][1]
    return b 

b1 = set_b(avg_corrs)
b2 = set_b(avg_corrs)
b1_rev = set_b(img1_corrs)
b2_rev = set_b(img2_corrs)

def solve_svd(A,b):
    U,s,Vh = svd(A)
    c = np.dot(U.T,b)
    w = np.dot(np.diag(1/s),c)
    z = np.dot(Vh.conj().T,w)
    return z

z1 = solve_svd(A1,b1)
z2 = solve_svd(A2,b2)
z1_rev = solve_svd(A_rev,b1_rev)
z2_rev = solve_svd(A_rev,b2_rev)

def phi_i(x_bar,i,corrs): #x is (x,y) and x_i is (x_i,y_i)
    if RBF == 1:
        eps = 0.0001
        x_i = corrs[i]
        x_i = np.asmatrix(x_i)
        x = np.asmatrix(x_bar)
        x_x_i = x - x_i
        norm = np.linalg.norm(x_x_i)
        norm_log = norm+eps
        output = norm**2 * np.log(norm_log)
    if RBF == 2:
        eps = 0.0001
        x_i = corrs[i]
        x_i = np.asmatrix(x_i)
        x = np.asmatrix(x_bar)
        x_x_i = x - x_i
        norm = np.linalg.norm(x_x_i)
        norm = norm **2 
        den = 2*(sigma**2)
        output = np.exp(-norm/den)
    return output

def set_T(x_,y_,z,img_corrs):

    p_2_x = float(z[num_corrs])
    p_1_x = float(z[num_corrs+1])
    p_0_x = float(z[num_corrs+2])
    p_2_y = float(z[num_corrs*2+3])
    p_1_y = float(z[num_corrs*2+4])
    p_0_y = float(z[num_corrs*2+5])
    T_x = 0
    T_y = 0
    
    for j in range(N):
        k_i_x = float(z[j])
        k_i_y = float(z[num_corrs+3+j])
            
        x_bar = [x_,y_]
        phi = phi_i(x_bar,j,img_corrs)
            
        T_x = T_x + k_i_x*phi   
        T_y = T_y + k_i_y*phi
    
    T_x = T_x + p_0_x + p_1_x*x_ + p_2_x*y_
    T_y = T_y + p_0_y + p_1_y*x_ + p_2_y*y_
    
    x_return = T_x
    y_return = T_y
    
    return y_return, x_return

#Forming Matrix of coordinates to be mapped 
T1 = np.array([[]]*W*H + [[1]],dtype=object)[:-1]
T1 = T1.reshape((H,W))
T2 = np.array([[]]*W*H + [[1]],dtype=object)[:-1]
T2 = T2.reshape((H,W))
T1_trans = np.array([[]]*W*H + [[1]],dtype=object)[:-1]
T1_trans = T1_trans.reshape((H,W))
T2_trans = np.array([[]]*W*H + [[1]],dtype=object)[:-1]
T2_trans = T2_trans.reshape((H,W))

for i in range(W):
    for j in range(H):
        T1[j,i] = [j,i]
        T2[j,i] = [j,i]
        T1_trans[j,i] = set_T(i,j,z1,img1_corrs)              #i is x, j is y
        T2_trans[j,i] = set_T(i,j,z2,img2_corrs)

max_1_x = T1_trans[0,0][1]
max_2_x = T2_trans[0,0][1]
min_1_x = T1_trans[0,0][1]
min_2_x = T2_trans[0,0][1]

max_1_y = T1_trans[0,0][0]
max_2_y = T2_trans[0,0][0]
min_1_y = T1_trans[0,0][0]
min_2_y = T2_trans[0,0][0]
for i in range(W):
    for j in range(H):
        if T1_trans[j,i][0] > max_1_y:
            max_1_y = T1_trans[j,i][0]
        if T1_trans[j,i][0] < min_1_y:
            min_1_y = T1_trans[j,i][0]
        if T2_trans[j,i][0] > max_2_y:
            max_2_y = T2_trans[j,i][0]
        if T2_trans[j,i][0] < min_2_y:
            min_2_y = T2_trans[j,i][0]
            
        if T1_trans[j,i][1] > max_1_x:
            max_1_x = T1_trans[j,i][1]
        if T1_trans[j,i][1] < min_1_x:
            min_1_x = T1_trans[j,i][1]
        if T2_trans[j,i][1] > max_2_x:
            max_2_x = T2_trans[j,i][1]
        if T2_trans[j,i][1] < min_2_x:
            min_2_x = T2_trans[j,i][1]

if max_2_x > max_1_x:
    max_x = max_2_x
else:
    max_x = max_1_x

if max_2_y > max_1_y:
    max_y = max_2_y
else:
    max_y = max_1_y

if min_2_x < min_1_x:
    min_x = min_2_x
else: 
    min_x = min_1_x

if min_2_y < min_1_y:
    min_y = min_2_y
else:
    min_y = min_1_y

if min_x < 0:
    min_x_tmp = 0
if min_y < 0: 
    min_y_tmp = 0

max_x_canvas = max_x - min_x
max_y_canvas = max_y - min_y
    
    
max_x_tmp = int(np.ceil(max_x_canvas))
max_y_tmp = int(np.ceil(max_y_canvas))

Img_T = np.array([[]]*max_x_tmp*max_y_tmp + [[1]],dtype=object)[:-1]
Img_T = Img_T.reshape((max_y_tmp,max_x_tmp))
T1_rev = np.array([[]]*max_x_tmp*max_y_tmp + [[1]],dtype=object)[:-1]
T1_rev = T1_rev.reshape((max_y_tmp,max_x_tmp))
T2_rev = np.array([[]]*max_x_tmp*max_y_tmp + [[1]],dtype=object)[:-1]
T2_rev = T2_rev.reshape((max_y_tmp,max_x_tmp))
for i in range(max_x_tmp):
    for j in range(max_y_tmp):
        Img_T[j,i] = [j,i]
        Img_T[j,i] = [j,i]
        T1_rev[j,i] = set_T(i,j,z1_rev,avg_corrs)       #i is x, j is y
        T2_rev[j,i] = set_T(i,j,z2_rev,avg_corrs)
  
#attempt at interpolation for Image 1 
x_inter1 = np.asarray(list(range(W)))
y_inter1 = np.asarray(list(range(H)))
z_inter1 = img_list[0]           #I1 in that figure
x_new1 = np.zeros(max_x_tmp*max_y_tmp)
y_new1 = np.zeros(max_y_tmp*max_x_tmp)
f1 = interp2d(x_inter1,y_inter1,z_inter1,kind = 'cubic')
z_new1 = np.zeros((max_y_tmp,max_x_tmp))
idx = 0
for i in range(max_x_tmp):
    for j in range(max_y_tmp):
        x_new1[idx] = T1_rev[j,i][1]
        y_new1[idx] = T1_rev[j,i][0]
        z_new1[j,i] = f1(x_new1[idx],y_new1[idx])
        idx = idx+1

#attempt at interpolation for Image 2
x_inter2 = np.asarray(list(range(W)))
y_inter2 = np.asarray(list(range(H)))
z_inter2 = img_list[1]           #I1 in that figure
x_new2 = np.zeros(max_x_tmp*max_y_tmp)
y_new2 = np.zeros(max_y_tmp*max_x_tmp)
z_new2 = np.zeros((max_y_tmp,max_x_tmp))
canvas = np.zeros((max_y_tmp,max_x_tmp))
canvas_eq = np.zeros((max_y_tmp,max_x_tmp))
f2 = interp2d(x_inter2,y_inter2,z_inter2,kind = 'cubic')
idx = 0
for i in range(max_x_tmp):
    for j in range(max_y_tmp):
        x_new2[idx] = T2_rev[j,i][1]
        y_new2[idx] = T2_rev[j,i][0]
        z_new2[j,i] = f2(x_new2[idx],y_new2[idx])
        idx=idx+1

#Histogram Equalization on the images
z_new1_eq = equalize_hist(z_new1)
z_new2_eq = equalize_hist(z_new2)

t = 0.6
for i in range(max_x_tmp):
    for j in range(max_y_tmp):
        canvas[j,i] = (z_new1_eq[j,i]*(1-t)) + (z_new2_eq[j,i]*t)
        canvas_eq[j,i] = (z_new1_eq[j,i]*(1-t)) + (z_new2_eq[j,i]*t)

#plt.imshow(canvas,cmap='gray')
plt.imshow(canvas_eq,cmap='gray')
for i in range(num_corrs):
    plt.scatter(avg_corrs[i][0],avg_corrs[i][1],c='#ff7f0e')
print("The output image is {data['Output file']}")
plt.savefig(output_name)


        

