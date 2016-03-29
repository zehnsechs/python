import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
import heapq
import time

class Detector:

    # threshold
    t = 20.0

    #circle_pos
    p = np.array([(0,3),(1,3),(2,2),(3,1),
                      (3,0),(3,-1),(2,-2),(1,-3),
                      (0,-3),(-1,-3),(-2,-2),(-3,-1),
                      (-3,0),(-3,1),(-2,2),(-1,3)])

    pattern_size = 16
    K = 8
    HARRIS_K = 0.04
    block = 7
    edge_thresh = 4
    sigma = float(2)
    gaus_fact = float(1)/(math.sqrt(2*math.pi)*sigma)

    def detect(self,img,max_count = 500, nonMaxSur = True, slow = True, comp_angle = False):
        start_time = time.time()
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

        def harris_score((y,x)):
            r = self.block/2
            a = 0
            b = 0
            c = 0
            scale = float(1)/((1 << 2) * self.block * float(255))
            scale_sq_sq = scale * scale * scale * scale
            

            for v in range(-3,4):
                for u in range(-3,4):
                    weight = self.gaus_fact*(math.exp(-(v*v+u*u)/(2*self.sigma*self.sigma)))
                    x0 = x + u
                    y0 = y + v
                    Ix = (int(image[y0][x0+1]) - int(image[y0][x0-1]))*2 + (int(image[y0-1][x0+1]) - int(image[y0-1][x0-1])) + (int(image[y0+1][x0+1]) - int(image[y0+1][x0-1]))
                    Iy = (int(image[y0+1][x0])- int(image[y0-1][x0]))*2 + (int(image[y0+1][x0-1]) - int(image[y0-1][x0-1])) + (int(image[y0+1][x0+1])- int(image[y0-1][x0+1]))
                    a += weight*Ix*Ix
                    b += weight*Iy*Iy
                    c += weight*Ix*Iy
            response = (float(a) * float(b) - float(c) * float(c) - self.HARRIS_K * (float(a) + float(b)) * (float(a) + float(b)))*scale_sq_sq
            #print response
            return response


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
                    if (circle_pix_clas(y,x,i%16) == 1):
                        count += 1
                        if (count > self.K):
                            #print 1
                            return True
                    else:
                        count = 0

           # if c == -1:
            #    return coner_cand(y,x,1)

            return False


        for i in range(4,dim[0]-4):
            for j in range(4,dim[1]-4):
                v = int(image[i][j])
                state = high_speed_test(i,j)
                #print (i,j,state)
                if (state!= 0):
                    
                    if coner_cand(i,j,state):
                        corners.append((i,j,0,0))
                        scores[i][j] = harris_score((i,j))
                        #print scores[i][j]
        time_det = time.time()
        print 'corners detected. Took: ', time_det - start_time, ' s'
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
            for (y,x,r,a) in corners:
                """
                if ((scores[y][x] < scores[y-1][x]) | (scores[y][x] < scores[y-1][x-1]) | 
                    (scores[y][x] < scores[y][x-1]) | (scores[y][x] < scores[y+1][x-1]) |
                    (scores[y][x] < scores[y+1][x]) | (scores[y][x] < scores[y+1][x+1]) |
                    (scores[y][x] < scores[y][x+1]) | (scores[y][x] < scores[y-1][x+1])):
                    pass
                else:
                    result.append((y,x,scores[y][x],0))
                scores[y][x] += 1
                """

                if ((scores[y][x] > scores[y-1][x]) & (scores[y][x] > scores[y-1][x-1]) & 
                    (scores[y][x] > scores[y][x-1]) & (scores[y][x] > scores[y+1][x-1]) &
                    (scores[y][x] > scores[y+1][x]) & (scores[y][x] > scores[y+1][x+1]) &
                    (scores[y][x] > scores[y][x+1]) & (scores[y][x] > scores[y-1][x+1])):
                        result.append((y,x,scores[y][x],0)) 
                          
        else:
            result = corners
        time_nonmax = time.time()
        print 'nonMaxSurpresion done. Took: ', time_nonmax -time_det, ' s'
         
        u_max=[3,3,2,1]

        def angle((y,x,r,_)):
            m_01 = 0
            m_10 = 0

            for u in range(-3,4):
                m_10 += u*image[y][x+u]

            for v in range(1,4):
                v_sum = 0
                d = u_max[v]
                for u in range(-d,d+1):
                    val_plus = int(image[y+v][x+u])
                    val_minus = int(image[y-v][x+u])
                    v_sum += (val_plus - val_minus)
                    m_10 += u*(val_plus + val_minus)
                m_01 += v*v_sum
            a = math.atan2(float(m_01),float(m_10))
            if a < 0:
                a += math.pi*2
            a = a*360/(2*math.pi)
            return (y,x,r,a)

        if (comp_angle):
            result = map(lambda k: angle(k),result) 
            time_ang = time.time()
            print 'angles done. Took: ', time_ang - time_nonmax, 's'


        if (len(result) > max_count):
            result = heapq.nlargest(max_count,result,key=(lambda x: x[2]))

        end_time = time.time()
        print 'All done. Took: ', end_time - start_time, ' s'
        return result


