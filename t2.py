import numpy as np
b=np.array([[1.,2.,3.],[1,2,3],[1,2,3]])
print(b)
print(b.dtype)
b=np.fromfunction(lambda x, y: 10 * x + y, (5, 4))
print(b)
print(b[0:2,2:3])
# print(b[0:2,2:3,-1])
print(b[...,1])
for i in b:
    print(i)
c=b>3
b[c]=0
print(b)
