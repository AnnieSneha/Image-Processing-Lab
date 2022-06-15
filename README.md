# image-processing-lab
Program 1:Develop a program to display Grayscale image using read and write operations
   
 import cv2<br>
 img=cv2.imread('b1.jpg',0)<br>
 cv2.imshow('b1',img)<br>
 cv2.waitKey(0)<br>
 cv2.destroyAllWindows()<br>
  
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173816090-77f93d35-0f19-4bc4-b5bc-f990d38e3f4c.png)

 --------------------------------------------------------------------------------------------------------------------------- 
Program 2:Develop a program to display image using matplotlib
 
 import matplotlib.image as mping<br>
 import matplotlib.pyplot as plt<br>
 img=mping.imread('f2.jpg')<br>
 plt.imshow(img)<br>
 
 OUTPUT:<br>
 ![image](https://user-images.githubusercontent.com/97939284/173809538-19372b96-f0f6-49f8-bc2a-f60dd9ae31af.png) 
------------------------------------------------------------------------------------------------------------------------------
Program 3:Develop a program to perform linear transformation 

from PIL import Image<br>
img=Image.open('l1.jpg')<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173813194-5f2722a6-034c-4747-b1dc-07a53a384e08.png)

-------------------------------------------------------------------------------------------------------------------------------
Program 4:Develop a program to covert color string to RGB color values

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
Program 5:Develop a program to create image using colors<br>

from PIL import Image<br>
img=Image.new("RGB",(200,400),(0, 0, 255))<br>
img.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/173813696-f552419d-9b39-4595-bf01-92f689afc915.png)

---------------------------------------------------------------------------------------------------------------------------------
Program 6:Develop a program to visualise the image using various color spaces

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
Program 7:Write a program to display the image attributes<br>

from PIL import Image<br>
image=Image.open('p2.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close();<br>

OUTPUT:<br>
Filename: p2.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (474, 313)<br>
Width: 474<br>
Height: 313<br>
-------------------------------------------------------------------------------------------------------------------------------------
 
