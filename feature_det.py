import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

# threshold
t = 10

def detect(path):
	mage_file = cbook.get_sample_data(path)
	image = plt.imread(image_file)
	dim = image.shape
	print dim

	#pixel in circle centerd at x,y with index i (0-15) clockwise start at top 
	#return -1 -> brighter
	#		 1 -> darker
	#		 0 -> eq/ out of range
	def circle_pix(x,y,i):
		if (i-3) >= 0:
			result[0] = image[i-3][j] - ip
		if (i+3) < dim[0]
			result[1] = image[i+3][j] - ip
		if (j-3) >= 0:
			result[2] = image[i][j-3] - ip
		if (j+3) < dim[1]
			result[3] = image[i][j+3] - ip
		return t

	def high_speed_test(i,j):
		d = 0
		b = 0
		ip = image[i][j]
		result  = np.array([])
		

		for r in result :
			if r == 1:
				b+= 1
			if r == -1:
				d += 1
		return (b >= 3) or (d >= 3)	


	for i in range(2,dim[0]-2):
		for j in range(2,dim[1]-2):
			if high_speed_test(i,j):
				if coner_cand(i,j):