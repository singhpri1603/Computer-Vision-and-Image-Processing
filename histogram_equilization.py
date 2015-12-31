from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def plot(data, title):
    plot.i += 1
    plt.subplot(2,3,plot.i)
    plt.imshow(data)
    plt.title(title)
plot.i = 0

def plot_his(data, title):
    plot.i += 1
    plt.subplot(2,3,plot.i)
    plt.plot(data)
    plt.title(title)
plot.i = 0

img = Image.open('this4.png').convert('L')
img.save('img_BW2.png')

#### plotting original histogram ####
pixels=np.array(img, dtype=np.int64)
row= len(pixels)
col= len(pixels[0])
his=np.zeros((256))
for i in range(row):
    for j in range(col):
        his[pixels[i][j]]=his[pixels[i][j]]+1
        
plot_his(his, 'Original histogram')

#### plotting cumulative histogram ####
hiscum=np.zeros((256))
hiscum[0]=his[0]
for i in range(1,256):
    hiscum[i]=hiscum[i-1]+his[i]
    
plot_his(hiscum, 'Cumulative histogram')

#### finding transformtion function to stretch the histogram ####
transfunc=np.zeros((256))
for i in range(256):
    transfunc[i]= round((255.0/(row*col))*hiscum[i])
plot_his(transfunc,'transformation function')

npixels=np.zeros((row,col))
for i in range(row):
    for j in range(col):
        npixels[i][j]=transfunc[pixels[i][j]]

#### plotting new histogram ####
nhis=np.zeros((256))
for i in range(row):
    for j in range(col):
        nhis[npixels[i][j]]=nhis[npixels[i][j]]+1
plot_his(nhis,'new histogram')

plot(pixels,'Original')
plot(npixels,'new image')
plt.show()