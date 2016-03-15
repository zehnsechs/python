import feature_det as fd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook
import numpy as np

p = np.array([(0,3),(1,3),(2,2),(3,1),
                  (3,0),(3,-1),(2,-2),(1,-3),
                  (0,-3),(-1,-3),(-2,-2),(-3,-1),
                  (-3,0),(-3,1),(-2,2),(-1,3)])


def get_point_list(l):
    b = []
    a = []
    print l
    for (x,y) in l:
        a.append(x)
        b.append(y)
    return a,b

path = '/Users/cknierim/python/horse.png'

image_file = cbook.get_sample_data(path)
image = plt.imread(image_file)

test = fd.detect(image)
(la,lb) = get_point_list(test)

plt.imshow(image)

for (x,y) in test:
    ai = []
    bi = []
    for j in range(16):
        (xd,yd) = p[j]
        ai.append(x+xd)
        bi.append(y+yd)

    plt.plot(bi,ai) 

plt.plot(lb,la,'ro')
plt.show()

