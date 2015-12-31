from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

def plot(data, title):
    plot.i += 1
    plt.subplot(2,3,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
plot.i = 0

#### convolution #### 
def convolution(pic, kernel):
    pic=np.lib.pad(pic,((2,2),(2,2)),'constant', constant_values=(0))
    conpic=np.zeros((len(pic),len(pic[0])))
    x=len(kernel)
    y=len(kernel[0])
    stop=0
    for i in range(2,len(pic)-2):
        for j in range(2,len(pic[0])-2):
            temp=pic[i-(x/2):i+((x/2)+1),j-(y/2):j+((y/2)+1)]
            start=time.time()
            mulsum=0
            for k in range(len(kernel)):
                for l in range(len(kernel[0])):
                    mulsum=mulsum+kernel[k][l]*temp[k][l]
            
            stop=stop+time.time()-start
            
            conpic[i][j]=mulsum
            #print conpic[i][j]
    conpic=conpic[2:-2,2:-2]
    return conpic, stop

img=Image.open('lena_gray.png')
pixels=np.array(img, dtype=float)
plot(pixels, 'original')

kernelGxA= np.array(([[1],[2],[1]]), np.float32)
kernelGxA=np.flipud(np.fliplr(kernelGxA))

kernelGxB= np.array(([[-1, 0 ,1]]), np.float32)
kernelGxB=np.flipud(np.fliplr(kernelGxB))

GxA, stopxA=convolution(pixels, kernelGxA)
GxB, stopxB=convolution(GxA, kernelGxB)


plot(GxA, 'GxA')
plot(GxB, 'GxB')

kernelGyA= np.array(([[-1],[0],[1]]), np.float32)
kernelGyA=np.flipud(np.fliplr(kernelGyA))

kernelGyB= np.array(([[1,2,1]]), np.float32)
kernelGyB=np.flipud(np.fliplr(kernelGyB))

GyA, stopyA=convolution(pixels, kernelGyA)
GyB, stopyB=convolution(GyA, kernelGyB)

plot(GyA, 'GyA')
plot(GyB, 'GyB')

G=(GxB**2 + GyB**2)**0.5
plot(G, 'G')


print stopxA+stopxB+stopyA+stopyB
plt.show()