import numpy as np 
from skimage import transform as tf


class Descriptor:   
    size = 16
    patch_size = 48
    threshold = 5
    kernel = 9
    key_points = 
[[
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


    def describe(self,features,img):
        patch_rad = selfpatch_size/2
        nr_kp = len(features)
        int_img = tf.integral_image(img)
        i = 0
        (heigth,width) = img.shape
        img = img
        half_ker = kernel/2

        def filter_fp():
            if(width < self.patch_size+self.kernel-1) or (heigth < self.patch_size+self.kernel-1):
                features = []
            for (fy,fx,res) in features:
                if (fx < patch_rad) | (fx > heigth-(self.patch_rad)+1) | (fy < self.patch_rad) | (fy > width-(self.patch_rad+1)):
                    features.remove((fy,fx,res))

        def sum(y,x):
            img_x = fx + x
            img_y = fy + y
            return tf.integrate(int_img,img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1, img_y - HALF_KERNEL + 1, img_x - HALF_KERNEL + 1) 

        filter_fp()
        result = np.zeros((len(features,self.size),dtype = int16)
        for (p,(fy,fx,res)) in list(enumerate(features)):
            for i in range(16):
                kp = self.key_points[i]
                for j in range(8):
                    [x1,y1,x2,y2] = kp[j]
                    result[p][i] += np.leftshift(sum(y1,x1) < sum(y2,x2),(7-i))
        return features , np.delete(result,range(len(features),nr_kp),0)

    def match(self,feat1,des1,feat2,des2):
        def dist(v1,v2):
            d = 0
            for k in range(self.size):
                if (v1[k] != v2[k]):
                    d +=1
            return d

        result = []
        for i in range(len(des1)):
            if des1[i].any() :
                for j in range(len(des2)):
                    d = dist(des1[i],des2[j])
                    if  d < self.threshold :
                        result.append((feat1[i],feat2[j],d))
                        print d
        return result

    
 





