import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook
import mpl_toolkits.axes_grid1 as mplt_a
import numpy as np

bw = False

a = np.zeros((10,10))
for i in range(10):
	for j in range(10):
		a[i][j] = (i+j)*0.01


print a[0][0]

#file = input("Bild")
#path = '/Users/cknierim/python/' + file
#print path
#image_file = cbook.get_sample_data(path)
#image = plt.imread(image_file)
image = a
dim = image.shape
print dim

if len(dim) == 2:
	bw = True

x1o = 0
y1o = 0

x2o = 0
y2o = dim[0]-1

x3o = dim[1]-1
y3o = dim[0]-1

x4o = dim[1]-1
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

u, s, vt = np.linalg.svd(am)

#test svd
S = np.zeros((8,9))
S[:8,:8] = np.diag(s)
print np.allclose(am, np.dot(u, np.dot(S, vt)))

print type(vt)

print s

hv = np.array(vt[8])

h = np.array([hv[0:3],hv[3:6],hv[6:9]])
print h.shape

width = max([x1n,x2n,x3n,x4n])+1 #- min([x1n,x2n,x3n,x4n])
heigth = max([y1n,y2n,y3n,y4n])+1 #- min([y1n,y2n,y3n,y4n])

print ":"

if bw:
	newimg = np.zeros((heigth,width))
else:
	newimg = np.zeros((heigth,width,4))

hom_p1n = np.dot(h,[x1o,y1o,1])
hom_p2n = np.dot(h,[x2o,y2o,1])
hom_p3n = np.dot(h,[x3o,y3o,1]) 
hom_p4n = np.dot(h,[x4o,y4o,1])



print "x1: " ,hom_p1n[0]/hom_p1n[2],x1n
print "y1: " , hom_p1n[1]/hom_p1n[2],y1n
print "x2: " , hom_p2n[0]/hom_p2n[2],x2n
print "y2: " , hom_p2n[1]/hom_p2n[2],y2n
print "x3: " , hom_p3n[0]/hom_p3n[2],x3n
print "y3: " , hom_p3n[1]/hom_p3n[2],y3n
print "x4: " , hom_p4n[0]/hom_p4n[2],x4n
print "y4: " , hom_p4n[1]/hom_p4n[2],y4n
print '---------------------'


invh = np.linalg.inv(h)


for i in range(heigth):
	for j in range(width):
				hom_p = np.dot(invh,[i,j,1])
				pos = (np.round(hom_p[0]/hom_p[2]),np.round(hom_p[1]/hom_p[2]))
				if pos[0] in range(dim[0]) and pos[1] in range(dim[1]):
					newimg[i][j] = image[pos[0]][pos[1]]

hom_p1o = np.dot(invh,[x1n,y1n,1])
hom_p2o = np.dot(invh,[x2n,y2n,1])
hom_p3o = np.dot(invh,[x3n,y3n,1]) 
hom_p4o = np.dot(invh,[x4n,y4n,1])


print "x1: " , hom_p1o[0]/hom_p1o[2],x1o
print "y1: " , hom_p1o[1]/hom_p1o[2],y1o
print "x2: " , hom_p2o[0]/hom_p2o[2],x2o
print "y2: " , hom_p2o[1]/hom_p2o[2],y2o
print "x3: " , hom_p3o[0]/hom_p3o[2],x3o
print "y3: " , hom_p3o[1]/hom_p3o[2],y3o
print "x4: " , hom_p4o[0]/hom_p4o[2],x4o
print "y4: " , hom_p4o[1]/hom_p4o[2],y4o



def back_trans((x1,x2)):
	hom_pt = np.dot(invh,[x1,x2,1])
	return hom_pt[0]/hom_pt[2], hom_pt[1]/hom_pt[2]

fig , (ax1,ax2) = plt.subplots(2,sharey = True,sharex = True)

print newimg.shape
print newimg[0][0]
ax1.imshow(image)
ax2.imshow(newimg)

mido = (sum([x1o,x2o,x3o,x4o])/4.,sum([y1o,y2o,y3o,y4o])/4.)
midn = (sum([x1n,x2n,x3n,x4n])/4.,sum([y1n,y2n,y3n,y4n])/4.)

ax1.plot([x1o,x2o,x3o,x4o],[y1o,y2o,y3o,y4o],'ro')
ax1.plot(mido[0],mido[1],'ys')
ax1.plot(back_trans(midn)[0],back_trans(midn)[1],'gs')

ax2.plot([x1n,x2n,x3n,x4n],[y1n,y2n,y3n,y4n],'bo')
ax2.plot(midn[0],midn[1],'gs')
plt.show()


"""
Fancy experiment :(sad)
"""
