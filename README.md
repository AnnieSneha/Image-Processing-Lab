# image-processing-lab
Program 1:Develop a program to display Grayscale image using read and write operations
   
   import cv2
   img=cv2.imread('b1.jpg',0)
   cv2.imshow('b1',img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
  OUTPUT:
  
Program 2:Develop a program to display image using matplotlib
  
  
  import matplotlib.image as mping
  import matplotlib.pyplot as plt
  img=mping.imread('f2.jpg')
  plt.imshow(img)
   OUTPUT:
