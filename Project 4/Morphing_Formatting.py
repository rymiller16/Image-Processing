#Ryan Miller
#u1067596
#Image Processing 
#Dr. Whitaker 

import json
from skimage import io
from skimage import color

def Morphing_Formatting(json_input):
    
    with open(json_input) as f:
        data = json.load(f)

    # print(f"The images are {data['Input files']}")
    corrs = data["Correspondences"]
    # print(f"There are {len(corrs)} sets of correspondences")
    # for i in range(len(corrs)):
    #     print(f"There are {len(corrs[i][0][1])} correspdondences between image {corrs[i][0][0]} and image {corrs[i][1][0]}")
    # print(f"The output file is {data['Output file']}")

    #extracting images
    num_images = len(data['Input files'])
    img_names = []
    img_list = []
    for i in range(num_images):
        img_names.append(data['Input files'][i])
        img_list.append(color.rgb2gray(io.imread(img_names[i])))

    num_set_corr = len(corrs)
    num_corrs = len(corrs[0][0][1])
    all_corrs = []
    for i in range(num_images):
        all_corrs.append(corrs[0][i][1])

    output_name = data['Output file']
    num_outputs = data['Number ouputs']

    return num_images, img_names, img_list, num_set_corr, num_corrs, all_corrs, output_name, num_outputs
    
    


    

