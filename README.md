
# Program 1:Develop a program to display Grayscale image using read and write operations
   
 import cv2<br>
 img=cv2.imread('b1.jpg',0)<br>
 cv2.imshow('b1',img)<br>
 cv2.waitKey(0)<br>
 cv2.destroyAllWindows()<br>
  
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/175284445-95473665-f4e5-4a1b-a2f9-3d2c29b8bb02.png)

# Program 2:Develop a program to display image using matplotlib
 
 import matplotlib.image as mping<br>
 import matplotlib.pyplot as plt<br>
 img=mping.imread('f2.jpg')<br>
 plt.imshow(img)<br>
 
 OUTPUT:<br>
 ![image](https://user-images.githubusercontent.com/97939284/173809538-19372b96-f0f6-49f8-bc2a-f60dd9ae31af.png) 
 
# Program 3:Develop a program to perform linear transformation 

from PIL import Image<br>
img=Image.open('l1.jpg')<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173813194-5f2722a6-034c-4747-b1dc-07a53a384e08.png)

# Program 4:Develop a program to covert color string to RGB color values

from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
img3=ImageColor.getrgb("pink")<br>
print(img3)<br>
img4=ImageColor.getrgb("blue")<br>
print(img4)<br>

OUTPUT:<br>
(255, 255, 0)<br>
(255, 0, 0)<br>
(255, 192, 203)<br>
(0, 0, 255)<br>

# Program 5:Develop a program to create image using colors<br>

from PIL import Image<br>
img=Image.new("RGB",(200,400),(0, 0, 255))<br>
img.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173813696-f552419d-9b39-4595-bf01-92f689afc915.png)

# Program 6:Develop a program to visualise the image using various color spaces

import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('01.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178466981-5f93c893-e6e2-4e74-b20b-f832c3962fc6.png)<br>
![image](https://user-images.githubusercontent.com/97939284/178467063-f92ad28d-4ca7-4d86-9c80-af519243386d.png)<br>
![image](https://user-images.githubusercontent.com/97939284/178467093-fd13ec3f-526a-4b94-8615-10eaf0898406.png)<br>

# Program 7:Write a program to display the image attributes<br>

from PIL import Image<br>
image=Image.open('18.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>

OUTPUT:<br>
Filename: 18.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
size: (564, 1004)<br>
Width: 564<br>
Height: 1004<br>

# Program 8: Convert original image to gray scale and then binary
import cv2<br>
#read the image file<br>
img=cv2.imread('b2.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
#gray scale<br>
img=cv2.imread('b2.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
#Binary image<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/175278709-775c6722-555e-4a93-8e65-825ab6d4d318.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175278786-0595d26f-eb31-4ea0-9381-6891c9e5228b.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175278901-a86de8c1-2293-4c1c-8cfc-c4bbdd39d134.png)<br>

# Program 9:Resize the original image

import cv2
img=cv2.imread('b4.jpg')
print('original image length width',img.shape)
cv2.imshow('original image',img)
cv2.waitKey(0)
#to show the resized image
imgresize=cv2.resize(img,(150,160))
cv2.imshow('Resized image',imgresize) 
print('Resized image length width',imgresize.shape)
cv2.waitKey(0)

OUTPUT:<br>
original image length width (399, 600, 3)<br>
Resized image length width (160, 150, 3)<br>

![image](https://user-images.githubusercontent.com/97939284/175283009-9af4da4b-67f5-48b2-a1e0-294b87e8107d.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175283093-5cb847aa-8fc1-45a9-996b-59ba86584070.png)

# Program 10 :Write a program to read image using URL

from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://dpbfm6h358sh7.cloudfront.net/images/5391016/1165772661.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/175005167-37733182-adc1-4404-b89c-e98949093711.png)

# Program 11:Write a program to mask and blur the image

import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('Nemo.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97939284/175016581-f6306b36-20bf-4e24-bb37-078e583ec0a5.png)<br>

hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97939284/175018103-8f9c2618-3c9e-4b98-820a-aefe1a99c8e0.png)<br>

light_white=(0,0,200)<br>
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97939284/175021322-e0c791b4-b034-459b-872a-fdab3c014de8.png)

final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97939284/175022161-6a5274ee-c347-4a21-b126-471b2e687a6e.png)<br>

blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97939284/175022486-66bbc556-701b-4919-a3f5-f046379189e8.png)

# Program 12: Write a program to perform arithmatic operation on image

import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>

#Reading image files<br>
img1=cv2.imread('flo1.jpg')<br>
img2=cv2.imread('flo2.jpg')<br>

#Applying Numpy addition on images<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

#Saving the ouput image<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>

#saving the ouput image<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>

#Saving the output image<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>

#saving the output image<br>
cv2.imwrite('output.jpg',fimg4)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/175269380-a4bf888c-9b30-4add-ac24-f7c3cf7257c0.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175270010-a186dfe0-4797-4706-9453-c32c0f1fb01d.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175270050-30ce5312-883c-49d8-8789-75d5a3bdb533.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175270086-88159f3a-e0e8-425c-a57a-3725c409ff40.png)<br>

# Program 13: Develop the program to change the image to different color spaces

import cv2<br> 
img=cv2.imread('D:\\R.jpg')<br> 
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br> 
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br> 
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br> 
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br> 
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br> 
cv2.imshow("GRAY image",gray)<br> 
cv2.imshow("HSV image",hsv)<br> 
cv2.imshow("LAB image",lab)<br> 
cv2.imshow("HLS image",hls)<br> 
cv2.imshow("YUV image",yuv)<br> 
cv2.waitKey(0)<br> 
cv2.destroyAllWindows()<br> 

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/175273905-b150309d-e394-451e-b2e4-e89a55e3bd53.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175274121-4703590c-f8ac-4f15-b5de-e4ab77b04f9e.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175274354-a32fe70d-64c5-4e05-bf22-c5a14c9976b9.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175275129-a8bec00c-081e-41da-abff-164068546848.png)<br>
![image](https://user-images.githubusercontent.com/97939284/175274788-3ce92a66-1786-429d-88dc-601be8887b18.png)<br>

# Program 14 :Program to create an image using 2D array<br>

import cv2 as c<br>
import numpy as np<br>
from PIL import Image
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255, 192, 203]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/175276384-61662c7b-8a30-4104-b4ad-327a1a03b6e0.png)

# Program 15:Write a program using Bitwise operation

import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('3.jpg',1)<br>
image2=cv2.imread('3.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/176425095-2fb817b6-272c-4aab-9d45-15b9c8a4607d.png)<br>
![image](https://user-images.githubusercontent.com/97939284/176425327-3aa83223-2f3b-47b2-889c-2cbf9b42f162.png)

# Program 16:Develop a program using Blurring operation

#importing libraries<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('g.jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>
#Gaussian Blur<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow("Gaussian blurring",Gaussian)<br>
cv2.waitKey(0)<br>
#Median Blur<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
#Bilateral Blur<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/176425935-a0d040a6-c482-4603-a974-126837ad5b5e.png)
![image](https://user-images.githubusercontent.com/97939284/176426022-576171f7-a6b8-4892-a02a-1309901fd126.png)<br>
![image](https://user-images.githubusercontent.com/97939284/176426090-80765078-8b2d-4dfd-ac98-7d99ac345000.png)
![image](https://user-images.githubusercontent.com/97939284/176426151-957f20d2-6178-4d05-991b-60ac8fc17843.png)

# Program 17:Develop a program using Image Enhancement

from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('19.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(imag<br>)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>
 
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178441194-0a568592-6331-4b3a-8ed6-51ae0d5e9fbb.png)
![image](https://user-images.githubusercontent.com/97939284/178441265-f402310b-89de-4d9e-907f-cfb41c2cbbe5.png)
![image](https://user-images.githubusercontent.com/97939284/178441353-6fc39b70-449f-44ff-970a-0754d48a9eb7.png)
![image](https://user-images.githubusercontent.com/97939284/178441438-47e73cfd-8217-4a84-9544-f76160528a11.png)
![image](https://user-images.githubusercontent.com/97939284/178441521-12a4f349-13dc-468d-b68c-beb623456a33.png)

# Program 18:Devlop a program using Morphological Operation

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('18.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178465499-030463ff-990a-499f-a010-9f1eadcb418e.png)

# Program 19: Develop a program 1)Read the image 2) Write(save) the grayscale 3)Display the original image and grayscale

import cv2<br>
OriginalImg=cv2.imread('21.jpg')<br>
GrayImg=cv2.imread('21.jpg',0)<br>
isSaved=cv2.imwrite('‪E:/21.jpg',GrayImg)<br>
cv2.imshow("Display Original Image",OriginalImg)<br>
cv2.imshow("Display Grayscale Image",GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print("The image is successfully saved")<br>
   
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178717011-e082c3d6-35fd-48f1-b722-46a2bef27e26.png)<br>
![image](https://user-images.githubusercontent.com/97939284/178702666-e1b3e0c7-febc-4ea2-aa28-56d89c800593.png)
![image](https://user-images.githubusercontent.com/97939284/178717109-626afbc9-d80b-447d-ad8b-5093636e16c6.png)<br>
![image](https://user-images.githubusercontent.com/97939284/178717800-a64236d9-140f-4c0d-af28-565e3f187382.png)


# Program 20: Write a program to perform slicing with background<br>

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('20.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title("graylevel slicing with background")<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178706366-f113450f-736d-464a-ae80-1e8a166f2dde.png)

# Program 21: Write a program to perform slicing without background<br>                               

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('21.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title("graylevel slicing without background")<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178711059-a76f1541-9627-4c83-8473-7807388d6df3.png)

# HISTOGRAM
1)<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('20.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count') <br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178960550-41973ca5-ca1c-4aac-8edf-d68b221fc51a.png)

2)<br>
import numpy as np<br>
import cv2 as cv<br>
from matplotlib import pyplot as plt<br>
img = cv.imread('20.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv.imread('20.jpg',0)<br>
plt.hist(img.ravel(),256,[0,256]);<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178960592-cdb5b2fe-9ef6-4647-aff4-bf47c6f6733f.png)<br>
![image](https://user-images.githubusercontent.com/97939284/178960639-d553cb8a-437f-4c2d-bf77-dc8932fc4055.png)<br>

3)
from skimage import io<br>
import matplotlib.pyplot as plt<br>
img = io.imread('19.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
image = io.imread('19.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178960706-4d352ea4-4254-444d-8809-475a3ae5f6ca.png)<br><br>
![image](https://user-images.githubusercontent.com/97939284/178960716-3363ad10-a358-4084-9179-9fcf982c28e0.png)

4)<br><br>
import cv2<br><br>
import numpy as np<br><br>
img  = cv2.imread('23.jpg',0)<br><br>
hist = cv2.calcHist([img],[0],None,[256],[0,256])<br><br>
plt.hist(img.ravel(),256,[0,256])<br><br>
plt.show()<br><br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178960778-1c19d188-6c5a-42bc-b2ca-6740cf0eb206.png)<br><br>

5)<br><br>
import cv2  <br><br>
from matplotlib import pyplot as plt <br><br> 
img = cv2.imread('19.jpg',0) <br><br>
histr = cv2.calcHist([img],[0],None,[256],[0,256])<br><br> 
plt.plot(histr) <br><br>
plt.show()<br><br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178960881-74b31062-fb65-4f17-9316-f3ae831e8965.png)

6)
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('13.jpg')<br>
_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )<br>
_ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count')<br>
_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178960964-8e20ba58-4f19-4e33-96b8-a23ebee9770c.png)

7)
from matplotlib import pyplot as plt<br>
import numpy as np<br>
fig,ax = plt.subplots(1,1)<br>
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])<br>
ax.hist(a, bins = [0,25,50,75,100])<br>
ax.set_title("histogram of result")<br>
ax.set_xticks([0,25,50,75,100])<br>
ax.set_xlabel('marks')<br>
ax.set_ylabel('no. of students')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/178962442-17db314a-eae8-4cd3-a2d9-0c2863d729ea.png)



