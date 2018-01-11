"""
	Code by Tu Thanh Nguyen
	Date : September 17, 2017
	Last Update: September 18, 2017
"""

from PreProcessing.PreProcessing  import ScaleImage
import os
import os.path

PATH = 'Data-Raw/'
Validate = 'Validate/'
Labels = os.listdir(PATH)
SIZES = 512, 512
count = 0
for label in Labels:
	print "Resize image processing Label:", label 
	im = ScaleImage(PATH+label+'/')
	files = im.readFile()
	index=1
	if len(files) > 0:
		for file in files:
			print "Processing file ", file 
			img = im.readImage(file)
			img = im.resizeImage(img,SIZES)
			im.saveImage(Validate + label + '/' + str(index)+'.jpg',img)
			index = index + 1
			count = count + 1
print "Number of Fruit : ",count
