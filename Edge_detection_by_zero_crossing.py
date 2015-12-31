from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot(data, title):
    plot.i += 1
    plt.subplot(1,3,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
plot.i = 0

img=Image.open('result_col.png')
pixels=np.array(img, dtype=float)
plot(pixels, 'original')

#### convolution by Differentiation of Gaussian kernel ####
DoGkernel= np.array(([[0,0,-1,-1,-1,0,0], [0,-2,-3,-3,-3,-2,0], [-1,-3,5,5,5,-3,-1], [-1,-3,5,16,5,-3,-1], [-1,-3,5,5,5,-3,-1], [0,-2,-3,-3,-3,-2,0], [0,0,-1,-1,-1,0,0]]), np.float32)                        
DoGkernel= np.flipud(np.fliplr(DoGkernel))

#DoGImage=np.zeros((len(pixels),len(pixels[0])))
DoGImage=signal.convolve2d(pixels, DoGkernel,mode='full', boundary='fill', fillvalue=0)

#plot(DoGImage, 'DoG image')

DoGzcross= np.ones((len(DoGImage), len(DoGImage[0])))
DoGzcross2=np.ones((len(DoGImage), len(DoGImage[0])))

#### Finding zero-crossing ####
for i in range(len(DoGImage)-1):
    for j in range(len(DoGImage[0])-1):
        count=0
        lst=[]
        lst.append(DoGImage[i][j])
        lst.append(DoGImage[i+1][j])
        lst.append(DoGImage[i+1][j+1])
        lst.append(DoGImage[i][j+1])
        if DoGImage[i][j]<0:
            
            count=count+1
        if DoGImage[i+1][j]<0:
            
            count=count+1
        if DoGImage[i+1][j+1]<0:
            
            count=count+1
        if DoGImage[i][j+1]<0:
            
            count=count+1
            
        
        if count>0 & count<4:
            DoGzcross[i][j]=0
            
            #### thresholding zero-crossing ####
            if max(lst)-min(lst)>=400:
                DoGzcross2[i][j]=0

plot(DoGzcross,'Zero crossing in DoG')
#plot(DoGzcross2,'DoG')


#### convolution by Laplacian of Gaussian kernel ####
LoGkernel= np.array(([[0,0,1,0,0], [0,1,2,1,0], [1,2,-16,2,1], [0,1,2,1,0], [0,0,1,0,0]]), np.float32)                        
LoGkernel= np.flipud(np.fliplr(LoGkernel))

LoGImage=signal.convolve2d(pixels, LoGkernel,mode='full', boundary='fill', fillvalue=0)

#plot(LoGImage, 'LoG image')

LoGzcross= np.ones((len(LoGImage), len(LoGImage[0])))
LoGzcross2= np.ones((len(LoGImage), len(LoGImage[0])))

#### finding zero-crossing ####
for i in range(len(LoGImage)-1):
    for j in range(len(LoGImage[0])-1):
        count=0
        lst=[]
        lst.append(LoGImage[i][j])
        lst.append(LoGImage[i+1][j])
        lst.append(LoGImage[i+1][j+1])
        lst.append(LoGImage[i][j+1])
        if LoGImage[i][j]<0:
            count=count+1
        if LoGImage[i+1][j]<0:
            count=count+1
        if LoGImage[i+1][j+1]<0:
            count=count+1
        if LoGImage[i][j+1]<0:
            count=count+1
        if count>0 & count<4:
            LoGzcross[i][j]=0
            
            #### thresholding zero-crossing ####
            if max(lst)-min(lst)>=400:
                LoGzcross2[i][j]=0
            
plot(LoGzcross,'Zero crossing in LoG')
#plot(LoGzcross2,'LoG')

plt.show()