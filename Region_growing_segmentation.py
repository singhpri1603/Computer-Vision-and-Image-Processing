from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def plot(data, title):
    plot.i += 1
    plt.subplot(1,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
plot.i = 0

def scan_image(regions):
    for i in range(1,len(regions)-1):
        for j in range(1,len(regions[0])-1):
            if regions[i][j]==0:
                return i,j 
    return -1,-1
            
#### recursive method for labeling the regions... it has gone out of use because of stackoverflow error ####               
def find_labels(x,y,pixels,regions):
    global threshold
    regions[x][y]=1
    if regions[x-1][y]==0 and abs(pixels[x][y]-pixels[x-1][y])<threshold:
        print pixels[x][y]
        print pixels[x-1][y] 
        print abs(pixels[x][y]-pixels[x-1][y])
        regions[x-1][y]=1
        print 1
        find_labels(x-1,y,pixels,regions)
    if regions[x+1][y]==0 and abs(pixels[x][y]-pixels[x+1][y])<threshold:
        regions[x+1][y]=1
        print 2
        find_labels(x+1,y,pixels,regions)
    if regions[x][y-1]==0 and abs(pixels[x][y]-pixels[x][y-1])<threshold:
        regions[x][y-1]=1
        print 3
        find_labels(x,y-1,pixels,regions)
    if regions[x][y+1]==0 and abs(pixels[x][y]-pixels[x][y+1])<threshold:
        regions[x][y+1]=1
        print 4
        find_labels(x,y+1,pixels,regions)

    return

img=Image.open('Peppers.jpg')

pixels=np.array(img, dtype=float)
final=np.array(img, dtype=float)
#pixels=pixels[1:70,1:70]
#pixels=pixels[0:400,0:400]
#final=final[0:400,0:400]
regions= np.zeros((len(pixels), len(pixels[0])), dtype=int)
pixels=np.lib.pad(pixels,((1,1),(1,1)),'constant', constant_values=(1))
regions=np.lib.pad(regions,((1,1),(1,1)),'constant', constant_values=(1))
plot(pixels, 'original')

threshold=10
threshold2=12
x=1
y=1
count=1

while x!=-1:
    queuex=deque([x],9999)
    queuey=deque([y],9999)
    regions[x][y]=count
    a=x
    b=y
    
    #### labeling the regions ####
    while len(queuex)!=0 :
        #print queuex
        #a=queuex.popleft()
        #print a
        x=queuex.popleft()
        y=queuey.popleft()
        
        #y=queuey.popleft()
    
        if regions[x-1][y]==0 and abs(pixels[x][y]-pixels[x-1][y])<threshold and abs(pixels[a][b]-pixels[x-1][y])<threshold2: 
            queuex.append(x-1)
            queuey.append(y)
            regions[x-1][y]=count
            #print "a"
        if regions[x+1][y]==0 and abs(pixels[x][y]-pixels[x+1][y])<threshold and abs(pixels[a][b]-pixels[x+1][y])<threshold2:
            queuex.append(x+1)
            queuey.append(y)
            regions[x+1][y]=count
            #print "b"
        if regions[x][y+1]==0 and abs(pixels[x][y]-pixels[x][y+1])<threshold and abs(pixels[a][b]-pixels[x][y+1])<threshold2:
            queuex.append(x)
            queuey.append(y+1)
            regions[x][y+1]=count
            #print "c"
        if regions[x][y-1]==0 and abs(pixels[x][y]-pixels[x][y-1])<threshold and abs(pixels[a][b]-pixels[x][y-1])<threshold2:
            queuex.append(x)
            queuey.append(y-1)
            regions[x][y-1]=count
            #print "D"
    count=count+1
    x,y= scan_image(regions)

pixels=pixels[1:-1,1:-1]
regions=regions[1:-1,1:-1]

means=np.zeros((count,3))
adj_list={}
for i in range(1,count):
    adj_list[i]=[]


#### creating adjacency list and calculating mean ####
for i in range(len(regions)-1):
    for j in range(len(regions[0])-1):
        means[regions[i][j]][0]=means[regions[i][j]][0]+pixels[i][j]### sum of pixels
        means[regions[i][j]][1]=means[regions[i][j]][1]+1 ### number of pixels
        if regions[i][j]!=regions[i+1][j] and regions[i+1][j] not in adj_list[regions[i][j]]:
            adj_list[regions[i][j]].append(regions[i+1][j])
        if regions[i][j]!=regions[i][j+1] and regions[i][j+1] not in adj_list[regions[i][j]]:
            adj_list[regions[i][j]].append(regions[i][j+1])
 
for i in range(len(means)):
    means[i][2]=round((means[i][0]/means[i][1]) , 2)

def merge(a,b,regions,adj_list,means):
    counter=0
    for i in range(len(regions)):
        for j in range(len(regions[0])):
            if regions[i][j]==b:
                regions[i][j]=a
                counter=counter+1
    if counter>0:
        means[i][0]=means[i][0]+means[j][0]
        means[i][1]=means[i][1]+means[j][1]
        if means[i][1]==0:
            means[i][2]=0
        else:
            means[i][2]=round((means[i][0]/means[i][1]),2)
        adj_list[a]=adj_list[a]+adj_list[b]
        adj_list[b]=[]


     
#### merging the regions with means within threshold ####  
#threshold2=12

for iterate in range(5):
    for j in adj_list:              
        for i in adj_list[j]:
            if abs(means[i][2]-means[j][2])<threshold2:
                merge(j,i,regions,adj_list,means)

for i in range(1,len(adj_list)):
    if len(adj_list[i])==1:
        merge(i,adj_list[i][0],regions,adj_list,means)

length=len(adj_list)
print length

#### deleting empty lists ####
for i in range(1,length):
    if adj_list[i]==[]:
        del adj_list[i]
        
print len(adj_list)

for i in range(len(regions)-1):
    for j in range(len(regions[0])-1):
        if regions[i][j]!=regions[i+1][j]:
            final[i][j]=0
            final[i+1][j]=0
        if regions[i][j]!=regions[i][j+1]:
            final[i][j]=0
            final[i][j+1]=0
#find_labels(x,y,pixels,regions)    
#plot_his(regions,'ye')
plot(final,'regions segmentation')
plt.show()   