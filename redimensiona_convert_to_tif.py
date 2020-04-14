# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:00:00 2020

@author: lucas



É necessário trocar os caminhos e atulizar o formato da imagem caso haja mais de uma.
"""

 
from skimage import data, io, filters
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
from PIL import Image
import os, sys

def files_path04(path):
    filenames=[]
    for p, _, files in os.walk(os.path.abspath(path)):
        for file in files:
            filenames.append( os.path.join(p, file))
    return filenames


import pathlib


lucas = os.listdir('C:\\Users\\lucas\\Desktop\\Codes_TCC\\Testes_Unet\\unet\\data\\test\\')

lista2=[]
for nomefamoso in lucas:
    lista2.append('C:\\Users\\lucas\\Desktop\\Codes_TCC\\Testes_Unet\\unet\\data\\test\\'+nomefamoso)
    
newsize = (256, 256)     #por ser a entrada da inception
for i in lista2:
  
        
        im = Image.open(i)
        im = im.resize(newsize) 
        im = im.save(i) 



for i in lista2:
    
    img= io.imread(i)
    #print(img.shape)
    a = i.replace('jpg', 'tif') # trocar para png e jpeg se necessario
    os.remove(i)
    io.imsave(a,img)







