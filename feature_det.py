import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math

# threshold
t = 10.0

#circle_pos
p = np.array([(0,3),(1,3),(2,2),(3,1),
                  (3,0),(3,-1),(2,-2),(1,-3),
                  (0,-3),(-1,-3),(-2,-2),(-3,-1),
                  (-3,0),(-3,1),(-2,2),(-1,3)])

pattern_size = 16
K = 8

def detect(img, nonMaxSur = True):
    quater = pattern_size/4
    #print img
    image = np.array(img)
    #print image
    dim = image.shape 
    if len(dim) != 2:
        return []

    corners = []
    #pixel in circle centerd at x,y with index i (0-15) clockwise start at top 
    #return  1 -> brighter
    #		-1 -> darker
    #		 0 -> eq/ out of range
    def circle_pix_dif(y,x,i):
        (yd,xd) = p[i]
        col_d = image[y+yd][x+xd] - v
       # print 'd', col_d
        if  abs(col_d) <= t:
            return 0;
        else:
            return np.sign(col_d)
    


    def high_speed_test(y,x):
        d = 0
        b = 0
        p = []
        result  = np.array([circle_pix_dif(y,x,0),circle_pix_dif(y,x,quater),
            circle_pix_dif(y,x,2*quater),circle_pix_dif(y,x,3*quater)])
        for i in range(4):
            r =result[i]
            if (r not in p) & ((r == result[(i+1)%4] )| (r == result[(i-1)%4])):
                p.append(r)

        if -1 in p:
            return -1
        if 1 in p:
            return 1
        else:
            return 0

   
    def corner_score((y,x)):
        sum = 0
        for i in range(16):
            sum += abs(image[y][x] - v)
        return sum

    def dist((x1,y1),(x2,y2)):
        return math.sqrt((x1-x2)**2+(y1-y2)**2)



    def coner_cand(y,x,c):
        #print 'punkt',(y,x,c)
        count = 0
        #darker
       # if c == 0:
        #    c =1
        if(c == -1):
            for i in range(pattern_size+K+1):
                #print count
                if (circle_pix_dif(y,x,i%16) == -1):
                    count += 1
                    if (count > K):
                        return True
                else:
                    count = 0

        #brighter
        if (c == 1):
            for i in range(pattern_size+K+1):
                
                if (circle_pix_dif(y,x,i%16) == 1):
                    count += 1
                    if (count > K):
                        return True
                else:
                    count = 0
        #if c == 1:
         #   return coner_cand(y,x,-1)

        return False


    for i in range(3,dim[0]-3):
        for j in range(3,dim[1]-3):
            v = int(image[i][j])
            state = high_speed_test(i,j)
            #print (i,j,state)
            if (state!= 0):
                if coner_cand(i,j,state):
                    corners.append((i,j))
   # print corners
    #print '_______________________________'

    if nonMaxSur:
        i = 0
        """
        while i < len(corners)-1:
            c1 = corners[i]
            c2 = corners[i+1]
            if dist(c1,c2) < 10:
                if corner_score(c1) > corner_score(c2):
                    corners.remove(c2)
                else:
                    corners.remove(c1)
            else:
                i += 1
                """

        while i < len(corners)-1:
            j = i+1
            while j < len(corners):
                c1 = corners[i]
                c2 = corners[j]
                if dist(c1,c2) < 10:
                    if corner_score(c1) > corner_score(c2):
                        corners.remove(c2)
                    else:
                        corners.remove(c1)
                else:
                     j += 1
            i += 1

                
    return corners


