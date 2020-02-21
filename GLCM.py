import cv2
import os
import glob
import numpy as np
import csv
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

#img_dir = "D:\MajorProject\Images" # Enter Directory of all images
img_dir = "D:\\MajorProject\\dataset\\2_cataract"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []

for f1 in files:
	image = io.imread(f1)
	#image = io.imread('C:\\Users\\Abhishek Kamal\\Desktop\\image.jpg')
	#image =io.imread('D:\\pic1.png')
	gray = color.rgb2gray(image)
	image = img_as_ubyte(gray)
	data.append(image)

#bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
#inds = np.digitize(image, bins)

	# GLCM properties
def contrast_feature(matrix_coocurrence):
	contrast = greycoprops(matrix_coocurrence, 'contrast')
	return contrast[0][0]
	#return "Contrast = ", contrast

def dissimilarity_feature(matrix_coocurrence):
	dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
	return dissimilarity[0][0]
	#return "Dissimilarity = ", dissimilarity[0][0]

def homogeneity_feature(matrix_coocurrence):
	homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
	return homogeneity[0][0]
	#return "Homogeneity = ", homogeneity

def energy_feature(matrix_coocurrence):
	energy = greycoprops(matrix_coocurrence, 'energy')
	return energy[0][0]
	#return "Energy = ", energy

def correlation_feature(matrix_coocurrence):
	correlation = greycoprops(matrix_coocurrence, 'correlation')
	return correlation[0][0]
	#return "Correlation = ", correlation

def asm_feature(matrix_coocurrence):
	asm = greycoprops(matrix_coocurrence, 'ASM')
	return asm[0][0]
	#return "ASM = ", asm

def writeglcm():
	with open("D:\\MajorProject\\cat.txt","a") as myfile:
		myfile.write(str(contrast_feature(matrix_coocurrence))+ " " + str(dissimilarity_feature(matrix_coocurrence)) + " " + str(homogeneity_feature(matrix_coocurrence)) + " " + str(energy_feature(matrix_coocurrence)) + " " + str(correlation_feature(matrix_coocurrence)) + " " + str(0)+"\n")


c=1
for img in data:
	print(c)
	c=c+1
	max_value = img.max()+1
	matrix_coocurrence = greycomatrix(img, [1], [0], levels=max_value, normed=True, symmetric=True)
	writeglcm()
	print(contrast_feature(matrix_coocurrence))
	print(dissimilarity_feature(matrix_coocurrence))
	print(homogeneity_feature(matrix_coocurrence))
	print(energy_feature(matrix_coocurrence))
	print(correlation_feature(matrix_coocurrence))
	print(asm_feature(matrix_coocurrence))