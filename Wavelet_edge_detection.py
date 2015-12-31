from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
from scipy.misc import imsave

def plot(data, title):
    plot.i += 1
    plt.subplot(2,3,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
plot.i = 0

#####------Adding noise to the image------#####
def AddNoise(image, noise_type):
    
    #####------Adding Gaussian noise with mean=20 and sigma=40------#####
    if noise_type == "gauss":
        row,col= image.shape
        img=np.zeros((row,col))
        for i in range(row):
            for j in range(col):
                img[i][j]=image[i][j]
        mean = 20
        gauss = np.random.normal(mean,40,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = img + gauss
        return noisy
    
    #####------Adding 20% Salt and Pepper noise------#####    
    elif noise_type == "s&p":
        row,col = image.shape
        img=np.zeros((row,col))
        for i in range(row):
            for j in range(col):
                img[i][j]=image[i][j]
        s_vs_p = 0.5
        amount = 0.040
        out = img
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        return out

#####------Wavelet transform------#####    
def WaveletTransform(image, wavelet):
    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs
    
    #####------For db2 trimming down the image to maintain size consistancy------#####
    if wavelet=='db2':
        cA=cA[1:len(cA),1:len(cA)]
        cH=cH[1:len(cH),1:len(cH)]
        cV=cV[1:len(cV),1:len(cV)]
        cD=cD[1:len(cD),1:len(cD)]
    print 'len(cA) '+str(len(cA))
    print 'len(cH) '+str(len(cH))
    print 'len(cV) '+str(len(cV))
    print 'len(cD) '+str(len(cD))
    
    #####------Scaling the transformed image by 2------#####
    cA=cv2.pyrUp(cA)
    cH=cv2.pyrUp(cH)
    cV=cv2.pyrUp(cV)
    cD=cv2.pyrUp(cD)
   
    print 'len(cA)up '+str(len(cA))
    print 'len(cH)up '+str(len(cH))
    print 'len(cV)up '+str(len(cV))
    print 'len(cD)up '+str(len(cD))
   
    return cA,cH,cV,cD

#####------Scale Multiplication after wavelet transform------#####
def scaleMul(img, image,wavelet):
    print '1'
    cA1,cH1,cV1,cD1= WaveletTransform(image,wavelet)
    print '2'
    cA2,cH2,cV2,cD2= WaveletTransform(cA1,wavelet)
    print '3'
    cA3,cH3,cV3,cD3= WaveletTransform(cA2,wavelet)
    print '4'
    cA4,cH4,cV4,cD4= WaveletTransform(cA3,wavelet)
    
    cHH1=cv2.multiply(cH1,cH2)
    cVV1=cv2.multiply(cV1,cV2)
    cDD1=cv2.multiply(cD1,cD2)
    
    cHH2=cv2.multiply(cH2,cH3)
    cVV2=cv2.multiply(cV2,cV3)
    cDD2=cv2.multiply(cD2,cD3)
    
    cHH3=cv2.multiply(cH3,cH4)
    cVV3=cv2.multiply(cV3,cV4)
    cDD3=cv2.multiply(cD3,cD4)
    
    #####------Adding the horizontal, vertical and diagonal details to form a combined edge map------#####
    final1=cHH1+cVV1+cDD1
    final2=cHH2+cVV2+cDD2
    final3=cHH3+cVV3+cDD3
    
    imsave('results/'+img+'level_1and2.png',final1)
    imsave('results/'+img+'level_2and3.png',final2)
    imsave('results/'+img+'level_3and4.png',final3)
    #plot(final1,img+'level 1&2')
    #plot(final2,img+'level 2&3')
    #plot(final3,img+'level 3&4')

img1=Image.open('Image_Carriage.jpg')
pixels1=np.array(img1, dtype=float)
#plot(pixels1, 'Image 1')
img2=Image.open('Image_Lena.jpg')
pixels2=np.array(img2, dtype=float)
#plot(pixels2, 'Image 2')
img3=Image.open('Image_Peppers.jpg')
pixels3=np.array(img3, dtype=float)
#plot(pixels3, 'Image 3')

noisy11=AddNoise(pixels1, 'gauss')
noisy12=AddNoise(pixels2, 'gauss')
noisy13=AddNoise(pixels3, 'gauss')

#plot(noisy11, 'Image 1 Gaussian noise')
#plot(noisy12, 'Image 2 Gaussian noise')
#plot(noisy13, 'Image 3 Gaussian noise')

noisy21=AddNoise(pixels1, 's&p')
noisy22=AddNoise(pixels2, 's&p')
noisy23=AddNoise(pixels3, 's&p')

#plot(noisy21, 'Image 1 Salt and Pepper noise')
#plot(noisy22, 'Image 2 Salt and Pepper noise')
#plot(noisy23, 'Image 3 Salt and Pepper noise')

scaleMul('image 1 Gauss haar', noisy11, 'haar')
scaleMul('image 1 Gauss db2',noisy11, 'db2')
scaleMul('image 1 S&P haar',noisy21, 'haar')
scaleMul('image 1 S&P db2',noisy21, 'db2')
scaleMul('image 2 Gauss haar',noisy12, 'haar')
scaleMul('image 2 Gauss db2',noisy12, 'db2')
scaleMul('image 2 S&P haar',noisy22, 'haar')
scaleMul('image 2 S&P db2',noisy22, 'db2')
scaleMul('image 3 Gauss haar',noisy13, 'haar')
scaleMul('image 3 Gauss db2',noisy13, 'db2')
scaleMul('image 3 S&P haar',noisy23, 'haar')
scaleMul('image 3 S&P db2',noisy23, 'db2')
