import brief
import feature_det as fd
#import compare_opencv as ccv 
import cv2
import feature_det as fd
from matplotlib import pyplot as plt

path = '/Users/cknierim/python/lenagray.png'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

path2 = '/Users/cknierim/python/6.png'
img2 = cv2.imread(path2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

my_dect = fd.Detector()
my_points = my_dect.detect(gray)
my_points2 = my_dect.detect(gray2)


my_des = brief.Descriptor()
new_f, t = my_des.describe(my_points,gray,False)

print my_points,new_f,t
#print new_f,len(new_f)
# Initiate FAST detector
fast = cv2.FeatureDetector_create("FAST")

# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")
brief.setInt('bytes',16)

# find the keypoints with FAST
kp = fast.detect(gray,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(gray, kp)

print brief.getInt('bytes')
print des
#print [(k.pt[1],k.pt[0]) for k in kp],len(kp)

print (t == des).all()

def get_point_list(l):
    b = []
    a = []
    for (x,y,_) in l:
        a.append(x)
        b.append(y)
    return a,b

