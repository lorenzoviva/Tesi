import numpy as np

x = np.ceil(np.random.random(100)*256)
y = np.ceil(np.random.random(100)*256)

xi = ((np.log(x) / np.log(256)) * 256)
yi = ((np.log(y) / np.log(256)) * 256)

di = yi-xi
i = np.ceil(np.power(256,(di/256)))

xd = (np.power(256,x/256))
yd = (np.power(256,y/256))

dd = np.abs(xd - yd)
d = ((np.log(dd)/np.log(256)) * 256)

for l in range(100):
    print("x:" + str(x[l]) + "\ty:" + str(y[l]) + "\ti:" + str(i[l]) + "\td:" + str(d[l]))
pass

# import numpy as np
#
# x = np.ceil(np.random.random(100)*256)
# y = np.ceil(np.random.random(100)*256)
#
# xi = np.floor((np.log(x) / np.log(256)) * 256)
# yi = np.floor((np.log(y) / np.log(256)) * 256)
#
# di = yi-xi
# i = np.ceil(np.power(256,(di/256)))
#
# xd = np.floor(np.power(256,x/256))
# yd = np.floor(np.power(256,y/256))
#
# dd = np.abs(xd - yd)
# d = np.ceil((np.log(dd)/np.log(256)) * 256)
#
# for l in range(100):
#     print("x:" + str(x[l]) + "\ty:" + str(y[l]) + "\ti:" + str(i[l]) + "\td:" + str(d[l]))
# pass
