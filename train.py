import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.multiclass import OneVsOneClassifier

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
	#return kq
	histh = cv2.calcHist([h],[0],None,[256],[0,256])
	hists = cv2.calcHist([s],[0],None,[256],[0,256])
	feature=np.append(ht_mean,cv2.meanStdDev(h))
	feature=np.append(feature,cv2.meanStdDev(s))
	feature=np.append(feature,kurtosis(histh))
	feature=np.append(feature,kurtosis(hists))
	feature=np.append(feature,skew(histh))
	feature=np.append(feature,skew(hists))
	return feature

# load the training dataset
train_path  = "Train"
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
train_features = []
train_labels   = []

# loop over the training dataset
i = 1
print "[STATUS] Started extracting haralick textures.."
for train_name in train_names:
	cur_path = train_path + "/" + train_name
	cur_label = train_name
	i = 1
	for file in glob.glob(cur_path + "/*.jpg"):
		print "Processing Image - {} in {}".format(i, cur_label)
		# read the training image
		image = cv2.imread(file)
		img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

		# convert the image to grayscale
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# extract haralick texture from the image
		features = extract_features(img)
		# append the feature vector and label
		train_features.append(features)
		train_labels.append(cur_label)

		# show loop update
		i += 1		
# have a look at the size of our feature vector and labels
print "Training features: {}".format(np.array(train_features).shape)
print "Training labels: {}".format(np.array(train_labels).shape)

# create the classifier
print "[STATUS] Creating the classifier.."
#clf_svm = LinearSVC(random_state=9)
clf_svm = AdaBoostClassifier(SVC(probability=True,kernel='linear'),n_estimators=50, learning_rate=1.0, algorithm='SAMME')
#clf_svm=KNeighborsClassifier(n_neighbors=3)
#clf.fit(X, y)
#clf1 = SVC( probability=True)
#clf_svm = BaggingClassifier(base_estimator=clf1)
#clf_svm=OneVsOneClassifier(LinearSVC(random_state=0))
# fit the training data and labels
print "[STATUS] Fitting data/label to model.."
clf_svm.fit(train_features, train_labels)
joblib.dump(clf_svm, 'model-224.pkl') 
# loop over the test images
# test_path = "Test1"
# index=0
# for file in glob.glob(test_path + "/*.jpg"):
# 	# read the input image
# 	image = cv2.imread(file)

# 	# convert to grayscale
# 	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# 	# extract haralick texture from the image
# 	features = extract_features(img)

# 	# evaluate the model and predict label
# 	prediction = clf_svm.predict(features.reshape(1, -1))[0]

# 	# show the label
# 	cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
# 	print "Prediction - {}".format(prediction)

# 	# display the output image
# 	cv2.imwrite('KQ/' + str(index)+prediction+'.jpg',image)
# 	index=index+1
# 	cv2.waitKey(0)
