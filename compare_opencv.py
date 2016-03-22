import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cbook 
import feature_det as fd

path = '/Users/cknierim/python/13.png'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print gray.shape

# Initiate FAST object with default values
fast = cv2.FeatureDetector_create("FAST")

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = np.array(img)
cv2.drawKeypoints(img, kp,img2, color=(0,255,0))


list_kp = [(k.pt[1],k.pt[0],k.response) for k in kp]
print "Total Keypoints with nonmaxSuppression: ", len(kp),list_kp

cv2.imwrite('fast_true.png',img2)

list_kp = [(k.pt[1],k.pt[0]) for k in kp]


# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(gray,None)
list_kp2 = [(k.pt[1],k.pt[0],k.response) for k in kp]
print '########################################'
print "Total Keypoints without nonmaxSuppression: ", len(kp) ,list_kp2

print'_____________________________________________'

img3 = np.array(img) 
cv2.drawKeypoints(img, kp,img3, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)



def get_point_list(l):
    b = []
    a = []
    for (x,y,z) in l:
        a.append(x)
        b.append(y)
    return a,b
my_dect = fd.Detector()
my_points = my_dect.detect(gray,True,True)

print "Total Keypoints own impl with nonmaxSuppression: ", len(my_points) , my_points



(la,lb) = get_point_list(my_points)
plt.imshow(gray)
plt.plot(lb,la,'ro')
plt.savefig('my_fig.png')

points = []

for p in list_kp2:
    if p in my_points:
        my_points.remove(p)
    else:
        points.append(p)

"""
print 'nicht bei mir:---------------------'
for (y,x,z) in points:
    print 'Punkt' ,(y,x) , gray[y][x]
    for i in range(16):
        (yd,xd) = my_dect.p[i]
        print '     ',i, gray[y+yd][x+xd]
    print '------------------'
    


print 'nur bei mir: --------------------------------'
for (y,x,z) in my_points:
    print 'Punkt' ,(y,x) , gray[y][x]
       
    for i in range(16):
        (yd,xd) = my_dect.p[i]
        print '     ',i, gray[y+yd][x+xd]
    print '------------------'

"""


