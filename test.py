import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook
import mpl_toolkits.axes_grid1 as mplt_a
import numpy as np


image_file = cbook.get_sample_data('/Users/cknierim/Desktop/python/png1.png')
image = plt.imread(image_file)

dim = image.shape


x1o = 0
y1o = 0

x2o = 0
y2o = dim[0]

x3o = dim[1]
y3o = dim[0]

x4o = dim[1]
y4o = 0

x1n,y1n = input("P1") 
x2n,y2n = input("P2") 
x3n,y3n = input("P3") 
x4n,y4n = input("P4") 



ax1 = (-x1o,-y1o,-1,  0,0,0,        x1n*x1o,x1n*y1o,x1n)
ay1 = (0,0,0,         -x1o,-y1o,-1, y1n*x1o,y1n*y1o,y1n)


ax2 = (-x2o,-y2o,-1,  0,0,0,        x2n*x2o,x2n*y2o,x2n)
ay2 = (0,0,0,         -x2o,-y2o,-1, y2n*x2o,y2n*y2o,y2n)


ax3 = (-x3o,-y3o,-1,  0,0,0,        x3n*x3o,x3n*y3o,x3n)
ay3 = (0,0,0,        -x3o,-y3o,-1,  y3n*x3o,y3n*y3o,y3n)


ax4 = (-x4o,-y4o,-1,  0,0,0,        x4n*x4o,x4n*y4o,x4n)
ay4 = (0,0,0,         -x4o,-y4o,-1, y4n*x4o,y4n*y4o,y4n)


am = (ax1,ay1,ax2,ay2,ax3,ay3,ax4,ay4)

u, s, v = np.linalg.svd(am)

print type(v)

hv = np.array(v[8])

h = np.array([hv[0:3],hv[3:6],hv[6:9]])
print h.shape

width = max([x1n,x2n,x3n,x4n]) - min([x1n,x2n,x3n,x4n])
heigth = max([y1n,y2n,y3n,y4n]) - min([y1n,y2n,y3n,y4n])

print ":"

newimg = np.zeros((width,heigth,4))

hom_p1n = np.dot(h,[x1o,y1o,1])
hom_p2n = np.dot(h,[x2o,y2o,1])
hom_p3n = np.dot(h,[x3o,y3o,1]) 
hom_p4n = np.dot(h,[x4o,y4o,1])


"""
print "x1: " ,hom_p1n[0]/hom_p1n[2],x1n
print "y2: " , hom_p2n[1]/hom_p2n[2],y2n
print "x3: " , hom_p3n[0]/hom_p3n[2],x3n
print "y3: " , hom_p3n[1]/hom_p3n[2],y3n
print "x4: " , hom_p4n[0]/hom_p4n[2],x4n
print "y4: " , hom_p4n[1]/hom_p4n[2],y4n
"""

invh = np.linalg.inv(h)


for i in range(heigth):
	for j in range(width):
				hom_p = np.dot(invh,[i,j,1])
				pos = (np.round(hom_p[0]/hom_p[2]),np.round(hom_p[1]/hom_p[2]))
				if pos[0] in range(dim[0]) and pos[1] in range(dim[1]):
					newimg[i][j] = image[pos[0]][pos[1]]


a = np.zeros((100,100))
for i in range(0,100):
	for j in range(0,100):
		a[i][j] = (i+j)*0.01

print a[0][0]

fig , (ax1,ax2) = plt.subplots(2,sharey = True,sharex = True)

ax1.imshow(image)
ax2.imshow(newimg)

mid= ()


ax1.plot([x1o,x2o,x3o,x4o],[y1o,y2o,y3o,y4o],'ro')
ax1.plot(sum([x1o,x2o,x3o,x4o])/4.,sum([y1o,y2o,y3o,y4o])/4.,'gs')
ax2.plot([x1n,x2n,x3n,x4n],[y1n,y2n,y3n,y4n],'bo')
ax2.plot(sum([x1n,x2n,x3n,x4n])/4.,sum([y1n,y2n,y3n,y4n])/4.,'gs')
plt.show()