import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

# threshold
t = 10

#circle_pos
p = np.array([(0,3),(1,3),(2,2),(3,1),
                  (3,0),(3,-1),(2,-2),(1,-3),
                  (0,-3),(-1,-3),(-2,-2),(-3,-1),
                  (-3,0),(-3,1),(-2,2),(-1,3)])
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
        (xd,yd) = p[i]
        col_d = image[x+xd][y+yd] - image[x][y]
        if  abs(col_d) < t:
            return 0;
        else:
            result np.sign(col_d)
    


	def high_speed_test(i,j):
		d = 0
		b = 0
		result  = np.array([circle_pix(i,j,0),circle_pix(i,j,4)
                            circle_pix(i,j,8),circle_pix(i,j,12)])
		
		for r in result :
			if r == 1:
				b+= 1
			if r == -1:
				d += 1
		return (b >= 3) or (d >= 3)	


	for i in range(3,dim[0]-3):
		for j in range(2,dim[1]-2):
			if high_speed_test(i,j):
				if coner_cand(i,j):