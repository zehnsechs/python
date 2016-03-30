import cv2
import feature_match as fm
from matplotlib import pyplot as plt

path = '/Users/cknierim/python/7.png'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


path2 = '/Users/cknierim/python/8.png'
img2 = cv2.imread(path2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

my_match = fm.Matcher()
matching = my_match.match(gray,gray2)


#print "Total Keypoints FAST with nonmaxSuppression: ", len(kpf),list_kpf
print '------------------------'
#print "Total Keypoints own impl with nonmaxSuppression: ", len(my_points) , my_points, score_sum(my_points)#, score_sum(my_points)/len(my_points)
print matching

#lines = [[(x1,y1),(x2,y2)] for ((y1,x1,r1,a1),(y2,x2,r2,a2),d) in matching]
lx1 = []
ly1 = []

lx2 = []
ly2 = []
for  ((y1,x1,r1,a1),(y2,x2,r2,a2),d) in matching: 
    lx1.append(x1)
    ly1.append(y1)
    lx2.append(x2)
    ly2.append(y2)


#lineCol = collections.LineCollection(lines)

#dst = cv2.addWeighted(gray,0.5,gray2,0.5,0)

#print lineCol
fig1, (a1,a2) =plt.subplots(2,sharey = True,sharex = True)
a1.imshow(gray)
a2.imshow(gray2)
#fig.add_collection(lineCol)
a1.plot(lx1,ly1,'ro')
a2.plot(lx2,ly2,'bo')
 
for i in range(len(lx1)):
    a1.annotate(i,xy=(lx1[i],ly1[i]),xytext=(lx1[i],ly1[i]))  
    a2.annotate(i,xy=(lx2[i],ly2[i]),xytext=(lx2[i],ly2[i]))  

plt.savefig('my_fig_match.png')


