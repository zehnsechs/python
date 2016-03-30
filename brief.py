import numpy as np 
from skimage import transform as tf
import cv2
import math


class Descriptor:   
    size = 16
    patch_size = 48
    threshold = 10
    kernel = 9
    key_points = [[
[-1, -2, -1, 7],
[-1, -14, 3, -3],
[-2, 1, 2, 11],
[6, 1, -7, -10],
[2, 13, 0, -1],
[5, -14, -3, 5],
[8, -2, 4, 2],
[8, -11, 5, -15]],
[
[-23, -6, -9, 8],
[6, -12, 8, -10],
[-1, -3, 1, 8],
[6, 3, 6, 5],
[-6, -7, -5, 5],
[-2, 22, -8, -11],
[7, 14, 5, 8],
[14, -1, -14, -5]],
[
[9, -14, 0, 2],
[-3, 7, 6, 22],
[6, -6, -5, -8],
[9, -5, -1, 7],
[-7, -3, -18, -10],
[-5, 4, 11, 0],
[3, 2, 10, 9],
[3, -10, 9, 4]],
[
[12, 0, 19, -3],
[15, 1, -5, -11],
[-1, 14, 8, 7],
[-23, 7, 5, -5],
[-6, 0, 17, -10],
[-4, 13, -4, -3],
[1, -12, 2, -12],
[8, 0, 22, 3]],
[
[13, -13, -1, 3],
[17, -16, 10, 6],
[15, 7, 0, -5],
[-12, 2, -2, 19],
[-6, 3, -15, -4],
[3, 8, 14, 0],
[-11, 4, 5, 5],
[-7, 11, 1, 7]],
[
[12, 6, 3, 21],
[2, -3, 1, 14],
[1, 5, 11, -5],
[-17, 3, 2, -6],
[8, 6, -10, 5],
[-2, -14, 4, 0],
[-7, 5, 5, -6],
[4, 10, -7, 4]],
[
[0, 22, -18, 7],
[-3, -1, 18, 0],
[22, -4, 3, -5],
[-7, 1, -3, 2],
[-20, 19, -2, 17],
[-10, 3, 24, -8],
[-14, -5, 5, 7],
[12, -2, -15, -4]],
[
[12, 4, -19, 0],
[13, 20, 5, 3],
[-12, -8, 0, 5],
[6, -5, -11, -7],
[-11, 6, -22, -3],
[4, 15, 1, 10],
[-4, -7, -6, 15],
[10, 5, 24, 0]],
[
[6, 3, -2, 22],
[14, -13, -4, 4],
[8, -13, -22, -18],
[-1, -1, 3, -7],
[-12, -19, 3, 4],
[10, 8, -2, 13],
[-1, -6, -5, -6],
[-21, 2, 2, -3]],
[
[-7, 4, 16, 0],
[-5, -6, -1, -12],
[-1, 1, 18, 9],
[10, -7, 6, -11],
[3, 4, -7, 19],
[5, -18, 5, -4],
[0, 4, 4, -20],
[-11, 7, 12, 18]],
[
[17, -20, 7, -18],
[15, 2, -11, 19],
[6, -18, 3, -7],
[1, -4, 13, -14],
[3, 17, -8, 2],
[2, -7, 6, 1],
[-9, 17, 8, -2],
[-6, -8, 12, -1]],
[
[4, -2, 6, -1],
[7, -2, 8, 6],
[-1, -8, -9, -7],
[-9, 8, 0, 15],
[22, 0, -15, -4],
[-1, -14, -2, 3],
[-4, -7, -7, 17],
[-2, -8, -4, 9]],
[
[-7, 5, 7, 7],
[13, -5, 11, -8],
[-4, 11, 8, 0],
[-11, 5, -6, -9],
[-6, 2, -20, 3],
[2, -6, 10, 6],
[-6, -6, 7, -15],
[-3, -6, 1, 2]],
[
[0, 11, 2, -3],
[-12, 7, 5, 14],
[-7, 0, -1, -1],
[0, -16, 8, 6],
[11, 22, -3, 0],
[0, 19, -17, 5],
[-14, -23, -19, -13],
[10, -8, -2, -11]],
[
[6, -11, 13, -10],
[-7, 1, 0, 14],
[1, -12, -5, -5],
[7, 4, -1, 8],
[-5, -1, 2, 15],
[-1, -3, -10, 7],
[-6, 3, -18, 10],
[-13, -7, 10, -13]],
[
[-1, 1, -10, 13],
[14, -19, -14, 8],
[-13, -4, 1, 7],
[-2, 1, -7, 12],
[-5, 3, -5, 1],
[-2, -2, -10, 8],
[14, 2, 7, 8],
[9, 3, 2, 8]]]


    def describe(self,features,img,use_orient):
        patch_rad = self.patch_size/2
        nr_kp = len(features)
        (heigth,width) = img.shape
        int_img = tf.integral_image(img)
        #test = cv2.integral(img)
        like_cv_img = np.concatenate((np.array([[0]*(heigth+1)]).T,np.concatenate(([[0]*width],int_img),axis = 0)),axis = 1)
        i = 0
        #print (like_cv_img == test).all()
        like_cv_img = np.int32(like_cv_img)
        #print like_cv_img
        #print test
        print heigth,width
        print nr_kp
        half_ker = self.kernel/2
        print half_ker


        def sum(y,x,(s_an,c_an)):
            if use_orient:
                rx = int(float(x)*c_an - float(y)*s_an)
                ry = int(float(x)*s_an + float(y)*c_an)

                if (rx > 24): 
                    rx = 24 
                if (rx < -24):
                    rx = -24
                if (ry > 24):
                    ry = 24
                if (ry < -24):
                    ry = -24;
                x = rx
                y = ry

            img_x = int(fx+0.5 + x)
            img_y = int(fy+0.5 + y)
            #print fy,fx,img_y,img_x
            #print tf.integrate(int_img,img_y + half_ker , img_x + half_ker , img_y - half_ker +1, img_x - half_ker +1) 
            #print test.dtype ,test[img_y + half_ker + 1, img_x + half_ker + 1] ,  int(test[img_y + half_ker + 1, img_x - half_ker ]) , int(test[img_y - half_ker , img_x + half_ker + 1]) ,  int(test[img_y - half_ker , img_x - half_ker ])
            #print int_img.dtype ,int_img[img_y + half_ker + 1, img_x + half_ker + 1] ,  int(int_img[img_y + half_ker + 1, img_x - half_ker ]) , int(int_img[img_y - half_ker , img_x + half_ker + 1]) ,  int(int_img[img_y - half_ker , img_x - half_ker ])
            #print int(test[img_y + half_ker + 1, img_x + half_ker + 1]) -  int(test[img_y + half_ker + 1, img_x - half_ker ]) - int(test[img_y - half_ker , img_x + half_ker + 1]) +  int(test[img_y - half_ker , img_x - half_ker ])
            #print '-----------------------'
            return int(like_cv_img[img_y + half_ker + 1, img_x + half_ker + 1]) -  int(like_cv_img[img_y + half_ker + 1, img_x - half_ker ]) - int(like_cv_img[img_y - half_ker , img_x + half_ker + 1]) +  int(like_cv_img[img_y - half_ker , img_x - half_ker ])
        
        border = patch_rad+half_ker

        if(width < 2*border) or (heigth < 2*border):
            return [],[]

        new_feat = []
        for i in range(nr_kp):
            (fy,fx,res,an) = features[i]
            #print border ,fy,fx
            if (fx < border) | (fx > width-border-1) | (fy < border) | (fy > heigth-border-1):
              pass
            else:new_feat.append((fy,fx,res,an))




        print len(new_feat)
        result = np.zeros((len(new_feat),self.size),dtype = np.uint16)
        for (p,(fy,fx,res,an)) in list(enumerate(new_feat)):
            sin_an = 0
            cos_an = 0
            if use_orient:
                rad_an = an*math.pi/float(180)
                sin_an = math.sin(rad_an)
                cos_an = math.cos(rad_an)
           # print fy,fx
            for i in range(16):
                kp = self.key_points[i]
                #print kp
                for j in range(8):
                    [x1,y1,x2,y2] = kp[j]
                    #print p,i,j,sum(y1,x1) , sum(y2,x2)
                    result[p][i] += np.left_shift(sum(y1,x1,(sin_an,cos_an)) < sum(y2,x2,(sin_an,cos_an)),(7-j))

        return new_feat , np.delete(result,range(len(new_feat),nr_kp),0)

    def match(self,feat1,des1,feat2,des2):
        def dist(v1,v2):
            b = (np.unpackbits(v1.view(np.uint8)) == np.unpackbits(v2.view(np.uint8)))
            return sum(np.invert(b))
        print des1.dtype
        result = []
        
        scores = np.zeros((len(des1),len(des2)),dtype = np.float16)

        for i in range(len(des1)):
            for j in range(len(des2)):
                insert = True
                scores[i][j] = dist(des1[i],des2[j])
                

        print scores
        for i in range(len(des1)):
            for j in range(len(des2)):
                print i,j,scores[i][j]
                if  scores[i][j] < self.threshold :
                    print 'a0,i',np.argmin(scores, axis=0),'a1,j',np.argmin(scores, axis=1)
                    if np.argmin(scores, axis=0)[j] == i  and np.argmin(scores, axis=1)[i] == j:  
                        result.append((feat1[i],feat2[j],scores[i][j]))
                            
        return result 

    
 





