# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:34:51 2022

@author: camil
"""
import Parser2
import cv2 as cv
import os
import VideotoData
import numpy as np
import cmd
import Interactive_HSV

#Hola mundo
import matplotlib as plt
ncarpetas, nvideos, Threshold_value, areafilterFactor, rangesearch_radius, limite, fi, ff = Parser2.parser()

croppData,videos=VideotoData.Video_Crop(ncarpetas, nvideos)

low_H, high_H, low_S, high_S, low_V, high_V = Interactive_HSV.AutoHSV(videos[0][0],croppData[0][0])

if not os.path.exists("Data"):
    os.mkdir("Data")
#cv.imwrite('Data/'+str(ncarpeta+1)+'\\'+str(nvideo+1)+'\\' +'Inner_Contour'+str(f)+'_.png', threshold_image)
for nc in range(ncarpetas[0]):
    if not os.path.exists('Data/'+str(nc+1)):
        os.mkdir('Data/'+str(nc+1))
    for nv in range(nvideos[0]):
        if not os.path.exists('Data/'+str(nc+1)+'/'+str(nv+1)):
            os.mkdir('Data/'+str(nc+1)+'/'+str(nv+1))
        if not os.path.exists('Data/'+str(nc+1)+'/'+str(nv+1)+'/Inner_Contour'):
            os.mkdir('Data/'+str(nc+1)+'/'+str(nv+1)+'/Inner_Contour')
        if not os.path.exists('Data/'+str(nc+1)+'/'+str(nv+1)+'/Total_Contour'):
            os.mkdir('Data/'+str(nc+1)+'/'+str(nv+1)+'/Total_Contour') 
            
        
        #VideotoData.Video_Threshold(videos[nc][nv],croppData[nc][nv],nc,nv,low_H, high_H, low_S, high_S, low_V, high_V )
        
        





print(croppData)
print(videos)




















""" cv.imshow("cropped", cropped_first_image)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)
 """
 

"""  #si al compilar con pyinstaller y luego ejecutar no funciona opencv
 #installar sudo apt-get install libcblas-dev
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev 
sudo apt-get install libqtgui4 
sudo apt-get install libqt4-test
  """
 