import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
def imhist(im):
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)
def cumsum(h):
	return [sum(h[:i+1]) for i in range(len(h))]
def histeq(im):
	h = imhist(im)
	cdf = np.array(cumsum(h))
	sk = np.uint8(255 * cdf)
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)
	return Y , h, H, sk
img = np.uint8(mpimg.imread(r'he/he.tif')*255.0)
mpimg.imsave(r'he/he.jpg',img)
img = np.uint8(mpimg.imread(r'he/he.jpg')*255.0)
img = np.uint8((0.2126* img[:,:,0]) + \
  		np.uint8(0.7152 * img[:,:,1]) +\
			 np.uint8(0.0722 * img[:,:,2]))
new_img, h, new_h, sk = histeq(img)
plt.subplot(121)
plt.imshow(img)
plt.title('original image')
plt.set_cmap('gray')
plt.subplot(122)
plt.imshow(new_img)
plt.title('hist. equalized image')
plt.set_cmap('gray')
plt.show()
fig = plt.figure()
fig.add_subplot(221)
plt.plot(h)
plt.title('Original histogram')
fig.add_subplot(222)
plt.plot(new_h)
plt.title('New histogram')

fig.add_subplot(223)
plt.plot(sk)
plt.title('Transfer function') #transfer function

plt.show()