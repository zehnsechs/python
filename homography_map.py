import numpy as np

def calc_map(org,dest):
	size = min(len(org),len(dest))
	a = np.empty((size*2,9))

	for i in range(size):
		xio = org[i][0]
		yio = org[i][1]
		xin = dest[i][0]
		yin = dest[i][1]

		a[i*2]   = (-xio,-yio,-1,  0,0,0,        xin*xio,xin*yio,xin)
		a[i*2+1] = (0,0,0,         -xio,-yio,-1, yin*xio,yin*yio,yin)

	u, s, vt = np.linalg.svd(a)
	hv = np.array(vt[8])
	h = np.array([hv[0:3],hv[3:6],hv[6:9]])

	return h

def map_hom(org,h):
	return map (lambda (x,y): np.dot(h,[x,y,1]),org)

def inv_map_hom(dest,h):
	invh = np.linalg.inv(h)
	return map (lambda (x,y): np.dot(invh,[x,y,1]),dest)

def from_hom_coord(coord):
	return map (lambda (x,y,w) : (x/w,y/w), coord) 

def calc_inv(h):
	return np.linalg.inv(h)
