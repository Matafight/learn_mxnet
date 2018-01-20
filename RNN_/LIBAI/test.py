import mxnet.ndarray as nd

a = nd.zeros((3,4))

a[0,0] = 1
for f in range(3):
    for s in range(4):
            val = int(str(f)+str(s))
            a[f,s]=val

print(a)
b = a.reshape((-1,))
print(b)