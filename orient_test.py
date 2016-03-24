import cv2
import feature_det as fd
from matplotlib import pyplot as plt
import numpy as np

path = '/Users/cknierim/python/3.png'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB()
orb.setInt('edgeThreshold',0)
orb.setInt('patchSize', 7)
orb.setInt('nLevels',1)
orb.setInt('nFeatures',20)
kp = orb.detect(gray,None)
list_kp = [(k.pt[1],k.pt[0],k.response,k.angle) for k in kp]

my_dect = fd.Detector()
my_points = my_dect.detect(gray,20,True,True,True,True)


# Initiate FAST object with default values
fast = cv2.FeatureDetector_create("FAST")
fast.setInt('threshold',20)
# find and draw the keypoints
kpf = fast.detect(img,None)
list_kpf = [(k.pt[1],k.pt[0]) for k in kpf]


print "Total Keypoints ORB with nonmaxSuppression: ", len(kp),list_kp
print'------------------------'
#print "Total Keypoints FAST with nonmaxSuppression: ", len(kpf),list_kpf
print '------------------------'
print "Total Keypoints own impl with nonmaxSuppression: ", len(my_points) , my_points

def get_point_list(l):
    b = []
    a = []
    for (x,y,r,an) in l:
        a.append(x)
        b.append(y)
    return a,b

(la,lb) = get_point_list(my_points)
plt.imshow(gray)
plt.plot(lb,la,'ro')
plt.savefig('my_fig_or.png')

img3 = np.array(img) 
cv2.drawKeypoints(img, kp,img3, color=(255,0,0))

cv2.imwrite('fast_false_or.png',img3)