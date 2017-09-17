from PIL import Image
import os
import os.path
import re

SIZES = 100, 100
class ScaleImage:
	def __init__(self,path):
		self.path=path
	def readFile(self):
		return [x for x in os.listdir(self.path) if re.match('.*\.[gif|png|jpeg|jpg]', x)]
	def readImage(self,path):
		return Image.open(path).convert('RGB')
	def resizeImage(self,img):
		return img.resize(SIZES,Image.ANTIALIAS)
	def saveImage(self,path,img):
			temp.save(path,quality=195)