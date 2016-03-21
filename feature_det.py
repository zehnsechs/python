import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math

class Detector:

    # threshold
    t = 10.0

    #circle_pos
    p = np.array([(0,3),(1,3),(2,2),(3,1),
                      (3,0),(3,-1),(2,-2),(1,-3),
                      (0,-3),(-1,-3),(-2,-2),(-3,-1),
                      (-3,0),(-3,1),(-2,2),(-1,3)])

    pattern_size = 16
    K = 8

    def detect(self,img, nonMaxSur = True, slow = True):
        quater = self.pattern_size/4
        N = self.pattern_size+self.K+1
        #print img
        image = np.array(img)
        #print image
        dim = image.shape 
        scores = np.zeros(dim)

        if len(dim) != 2:
            return []

        corners = []
        #pixel in circle centerd at x,y with index i (0-15) clockwise start at top 
        #return  1 -> brighter
        #		-1 -> darker
        #		 0 -> eq/ out of range
        def circle_pix_clas(y,x,i):
            (yd,xd) = self.p[i]
            col_d = image[y+yd][x+xd] - v
           # print 'd', col_d
            if  abs(col_d) <= self.t:
                return 0;
            else:
                return np.sign(col_d)
        
        def circle_pix_dif(y,x,i):
            (yd,xd) = self.p[i]
            return image[y+yd][x+xd] - v 
          
        


        def high_speed_test(y,x):
            d = 0
            b = 0
            p = []
            result  = np.array([circle_pix_clas(y,x,0),circle_pix_clas(y,x,quater),
                circle_pix_clas(y,x,2*quater),circle_pix_clas(y,x,3*quater)])
            for i in range(4):
                r =result[i]
                if (r not in p) & ((r == result[(i+1)%4])| (r == result[(i-1)%4])):
                    p.append(r)

            if -1 in p:
                if slow:
                    if 1 in p:
                        return 2
                return -1
            if 1 in p:
                return 1
            else:
                return 0

       
        def corner_score((y,x)):
            #print (y,x,v)
            a0 = self.t
            d = np.empty((N))
            for k in range(N):
                d[k] = (circle_pix_dif(i,j,k%16))
            #print d
            for k in range(0,16,2):
                a = min(d[k+1],d[k+2])
                for l in range(3,self.K+1):
                    a = min(a,d[k+l])
               # print 'a',a 
                a0 = max(a0, min(a,d[k]))
                a0 = max(a0, min(a,d[k+self.K+1]))
            b0 = -a0
            for k in range(0,16,2):
                b = max(d[k+1],d[k+2])
                for l in range(3,self.K+1):
                     b = max(b,d[k+l])
                #print 'b',b 

                b0 = min(b0, max(b, d[k]))
                b0 = min(b0, max(b, d[k+self.K+1]))
            return -b0-1

        def coner_cand(y,x,c):
            #print 'punkt',(y,x,c)
            count = 0
            #darker
            #if c == 0:
             #   c =-1
            if(c == -1):
                for i in range(N):
                    #print count
                    #if (d[i] < -self.t):
                    if (circle_pix_clas(y,x,i%16) == -1):

                        count += 1
                        if (count > self.K):
                            return True
                    else:
                        count = 0

            #brighter
            elif (c == 1):
                for i in range(N):
                    
                    #if (d[i] > self.t):
                    if (circle_pix_clas(y,x,i%16) == 1):

                        count += 1
                        if (count > self.K):
                            return True
                    else:
                        count = 0

            elif (c == 2):
                #print (y,x)
                for i in range(N):
                    #print -1, count
                    if (circle_pix_clas(y,x,i%16) == -1):
                        count += 1
                        if (count > self.K):
                           # print -1
                            return True
                    else:
                        count = 0
                count = 0        

                for i in range(N):
                   # print 1, count
                    if (d[i] > self.t):
                        count += 1
                        if (count > self.K):
                            #print 1
                            return True
                    else:
                        count = 0

           # if c == -1:
            #    return coner_cand(y,x,1)

            return False


        for i in range(3,dim[0]-3):
            for j in range(3,dim[1]-3):
                v = int(image[i][j])
                state = high_speed_test(i,j)
                #print (i,j,state)
                if (state!= 0):
                    
                    if coner_cand(i,j,state):
                        corners.append((i,j,0))
                        scores[i][j] = corner_score((i,j))
                        #print scores[i][j]
       # print corners
        #print '_______________________________'

        print 'found' , len(corners)
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
            result = []
            #print scores
            for (y,x,_) in corners:
                if ((scores[y][x] < scores[y-1][x]) | (scores[y][x] < scores[y-1][x-1]) | 
                    (scores[y][x] < scores[y][x-1]) | (scores[y][x] < scores[y+1][x-1]) |
                    (scores[y][x] < scores[y+1][x]) | (scores[y][x] < scores[y+1][x+1]) |
                    (scores[y][x] < scores[y][x+1]) | (scores[y][x] < scores[y-1][x+1])):
                    pass
                else:
                    result.append((y,x,scores[y][x]))


                """((scores[y][x] >= scores[y-1][x]) & (scores[y][x] >= scores[y-1][x-1]) & 
                    (scores[y][x] >= scores[y][x-1]) & (scores[y][x] >= scores[y+1][x-1]) &
                    (scores[y][x] >= scores[y+1][x]) & (scores[y][x] >= scores[y+1][x+1]) &
                    (scores[y][x] >= scores[y][x+1]) & (scores[y][x] >= scores[y-1][x+1])):
                    result.append((y,x,scores[y][x])) """
                scores[y][x] += 1
        else:
            result = corners
                    
        return result


