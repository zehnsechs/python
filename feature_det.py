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

pattern_size = 16
K = 12

def detect(path):
    
    quater = pattern_size/4
	mage_file = cbook.get_sample_data(path)
	image = plt.imread(image_file)
	dim = image.shape
	print dim
    corners = []
    

	#pixel in circle centerd at x,y with index i (0-15) clockwise start at top 
	#return -1 -> brighter
	#		 1 -> darker
	#		 0 -> eq/ out of range
	def circle_pix_dif(x,y,i,v):
        (xd,yd) = p[i]
        col_d = image[x+xd][y+yd] - v
        if  abs(col_d) < t:
            return 0;
        else:
            result np.sign(col_d)
    


	def high_speed_test(i,j):
		d = 0
		b = 0
        v = image[x][y]
		result  = np.array([circle_pix_dif(i,j,0,v),circle_pix_dif(i,j,quater,v)
                            circle_pix_dif(i,j,2*quater,v),circle_pix_dif(i,j,3*quater,v)])
		for r in result :
			if r == 1:
				b+= 1
			if r == -1:
				d += 1
        if (b >= 3):
            return -1
        if (d >= 3):
            return 1
        else:
            return 0
    

    def coner_cand(x,y,c):
        count = 0
        v = image[x][y]
        
        #darker
        if(c == 1):
            for i in range(pattern_size+K+1):
                if (circle_pix_dif(x,y,i,v) == 1):
                    count += 1
                    if (count > K):
                        return True
                else:
                    count = 0

        #brighter
        if (c == -1):
            for i in range(pattern_size+K+1):
                    if (circle_pix_dif(x,y,i,v) == -1):
                        count += 1
                            if (count > K):
                                return True
                            else:
                                count = 0
        return False


	for i in range(3,dim[0]-3):
		for j in range(2,dim[1]-2):
            state = high_speed_test(i,j)
			if (state!= 0):
				if coner_cand(i,j,state):
                    corners.append((i,j))

    return corners

