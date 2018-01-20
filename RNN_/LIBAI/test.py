import mxnet.ndarray as nd

a = nd.zeros((3,4,5))

a[0,0,0] = 1
for f in range(3):
    for s in range(4):
        for t in range(5):
            val = int(str(f)+str(s)+str(t))
            a[f,s,t]=val

print(a)
b = a.reshape((-1,5))
print(b)