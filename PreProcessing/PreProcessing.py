"""
	Code written by Tu Thanh Nguyen
	Date : September 17, 2017 
	Last Update: September 18, 2017
"""

import os
import os.path
import re
import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from PIL import Image


class ScaleImage:
	def __init__(self,path):
		self.path=path
	def readFile(self):
		return [x for x in os.listdir(self.path) if re.match('.*\.[gif|png|jpeg|jpg|png|JPG]', x)]
	def readImage(self,im):
		return Image.open(self.path+im).convert('RGB')
	def resizeImage(self,img,SIZES):
		return img.resize(SIZES,Image.ANTIALIAS)
	def saveImage(self,path,img):
			img.save(path,quality=195)

class RectObject:
	"""
	"""
	def __init__(self,img):
		self.img=img
	def getRects(self):
		ret, threshed_img = cv2.threshold(self.img, 
			120, 255, cv2.THRESH_BINARY)
	    # find contours and get the external one
		image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
   		# list rects
   		# with e in rects 
   		#		e[0] = Acreage of rect
   		#		e[1] = coordinate point x
   		#		e[2] = coordinate point y
   		#		e[3] = width of rect
   		#		e[4] = hight of rect
		rects = []
		for c in contours:
			# get the bounding rect
			x, y, w, h = cv2.boundingRect(c)
			rects.append([w*h,x,y,w,h])

		return rects
	def getMaxRect(self):
		rects = self.getRects()
		max = rects[0]
		for e in rects:
			if e[0] > max [0]:
				max=e
		return max

