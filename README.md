
# Program 1:Develop a program to display Grayscale image using read and write operations
   
 import cv2<br>
 img=cv2.imread('b1.jpg',0)<br>
 cv2.imshow('b1',img)<br>
 cv2.waitKey(0)<br>
 cv2.destroyAllWindows()<br>
  
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173816090-77f93d35-0f19-4bc4-b5bc-f990d38e3f4c.png)

 --------------------------------------------------------------------------------------------------------------------------- 
# Program 2:Develop a program to display image using matplotlib
 
 import matplotlib.image as mping<br>
 import matplotlib.pyplot as plt<br>
 img=mping.imread('f2.jpg')<br>
 plt.imshow(img)<br>
 
 OUTPUT:<br>
 ![image](https://user-images.githubusercontent.com/97939284/173809538-19372b96-f0f6-49f8-bc2a-f60dd9ae31af.png) 
 
------------------------------------------------------------------------------------------------------------------------------
# Program 3:Develop a program to perform linear transformation 

from PIL import Image<br>
img=Image.open('l1.jpg')<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173813194-5f2722a6-034c-4747-b1dc-07a53a384e08.png)

-------------------------------------------------------------------------------------------------------------------------------
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

--------------------------------------------------------------------------------------------------------------------------------
# Program 5:Develop a program to create image using colors<br>

from PIL import Image<br>
img=Image.new("RGB",(200,400),(0, 0, 255))<br>
img.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173813696-f552419d-9b39-4595-bf01-92f689afc915.png)

---------------------------------------------------------------------------------------------------------------------------------
# Program 6:Develop a program to visualise the image using various color spaces

import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('b2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173814290-0aca2040-dcff-4ad5-8661-82f8265e52a7.png)<br>
![image](https://user-images.githubusercontent.com/97939284/173814439-9890ea4d-504b-4b0f-b1d5-24dbe52f9b18.png)<br>
![image](https://user-images.githubusercontent.com/97939284/173814560-3c54a329-6356-4a2d-884c-c6231bc1ac76.png)<br>

-------------------------------------------------------------------------------------------------------------------------------------
# Program 7:Write a program to display the image attributes<br>

import cv2<br><br>
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

OUTPUT:

------------------------------------------------------------------------------------------------------------------------------------------------
# Program 9:

import cv2<br>
img=cv2.imread('f1.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
#to show the resized image<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)
print('Resized image length width',imagresize.shape)<br>
cv2.waitKey(0)<br>

OUTPUT:

# Program 10 :Write a program to read image using URL

from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://dpbfm6h358sh7.cloudfront.net/images/5391016/1165772661.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/175005167-37733182-adc1-4404-b89c-e98949093711.png)


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
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()

