Read me for part 1 on Project 2 - Image Processing Dr. Whitaker. 

In order to try and prevent hard coding image inputs etc in the running on my python scripts, I have decided to take inputs from the user to determine the output.
There are a total of 5 images used in the analysis for first portion. When running the script, python will ask the user which picture they want analyzed. 

1 - Jerry Garcia 
2 - Jim Morrison
3 - Mark Knopfler
4 - Neil Young 
5 - Mick Jagger 

After choosing the picture to be analyzed (1-5), the script will then ask the user which type of filter they would like to apply. The options for this are: 

1 - linear 
2 - non-linear 

In total, there are 7 different linear and 7 different non-linear filters. The linear filters are: box filter, derivative filter, wiener filter, 2 different gaussian filters
(with different sigma values), a rectangular filter, and a combination of derivative filters. The non-linear filters are: median filter, bilateral filter, 
two non-local filters (using fast algorithim and slow algorithim), denoise tv chambolle filter, denoise tv bregman filter, and the denoise wavelet filter. 

After selecting the type of filtering to be analyzed, the script will then ask what type of noise the user would like to analyze the filter on. The output of the 
script will be a 3x3 figure with 9 total subplots. The original "ground" image will be presented, followed by the noisy image (using the noise type selected
by the user), followed by each of the 7 filter outputs. When I analyze the filters in my report, I will go in and change parameters. I decided not to take parameter inputs
from the user because there would be too many inputs to keep track of for each of the 14 total different types of filters. 

Enjoy. 