import feature_det as fd
import brief
import numpy as np
import time

class Matcher:
    blocksize = 20
    nr_feat = 10

    def match(self,img1,img2):
        start = time.time()

        detector = fd.Detector()
        descriptor = brief.Descriptor()

        kp1 = detector.detect(img1,self.nr_feat)
        kp2 = detector.detect(img2,self.nr_feat)or()

        detected = time.time()
        print 'Detected in ',detected - start,'s'

        new_kp1 , des1 = descriptor.describe(kp1,img1,False)
        new_kp2 , des2 = descriptor.describe(kp2,img2,False)

        described = time.time()
        print 'Described in ',described-detected,'s'

        matching = descriptor.match(new_kp1,des1,new_kp2,des2)

        matched = time.time()
        print 'Matched in ',matched - described,'s'

        return matching

 