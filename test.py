import numpy as np
from Hong_FastApEn import *
from Hong_FastCApEn import *
if __name__ == "__main__":
    import time
    import random
    U = np.array([random.randint(0, 100) for i in range(1000)])
    G = np.array([random.randint(0, 100) for i in range(1000)])
    ap = NewApEn(2, 0.2)
    ap1 = NewHuApEn(2, 0.2)
    t = time.time()
    print(ap.jinshishang(U))
    t1 = time.time()
    print(ap.hongbo_jinshishang(U))
    t2 = time.time()
    print(ap1.hujinshishang(U, G))
    t3 = time.time()
    print(ap1.hongbo_hujinshishang(U, G))
    t4 = time.time()
    print(t1-t)
    print(t2-t1)
    print(t3-t2)
    print(t4-t3)