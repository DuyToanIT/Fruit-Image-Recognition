import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

def extract_features(image):
	# calculate haralick texture features for 4 types of adjacency
	image=cv2.resize(image,(224,224))
	h=image[:,:,0]
	s=image[:,:,1]
	v=image[:,:,2]
	blur = cv2.GaussianBlur(s,(5,5),0)
	ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	h=h*th3
	v=v*th3
	s=s*th3
	textures = mt.features.haralick(v)
	# take the mean of it and return it
	ht_mean  = textures.mean(axis=0)
	histh = cv2.calcHist([h],[0],None,[256],[0,256])
	hists = cv2.calcHist([s],[0],None,[256],[0,256])
	feature=np.append(ht_mean,cv2.meanStdDev(h))
	feature=np.append(feature,cv2.meanStdDev(s))
	feature=np.append(feature,kurtosis(histh))
	feature=np.append(feature,kurtosis(hists))
	feature=np.append(feature,skew(histh))
	feature=np.append(feature,skew(hists))
	return feature

clf_svm = joblib.load('model-224.pkl') 
test_path = "Test"
index=0
for file in glob.glob(test_path + "/*.jpg"):
	# read the input image
	image = cv2.imread(file)

	# convert to grayscale
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	# extract haralick texture from the image
	features = extract_features(img)

	# evaluate the model and predict label
	prediction = clf_svm.predict(features.reshape(1, -1))[0]

	# show the label
	cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
	print "Prediction - {}".format(prediction)

	# display the output image
	cv2.imwrite('KQ/' + str(index)+prediction+'.jpg',image)
	index=index+1
	cv2.waitKey(0)
