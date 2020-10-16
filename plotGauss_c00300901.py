#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 22:49:32 2017

@author: maida

Plot bivariate Gaussian
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import math

#for question 3,4,5,6,7 . To run the provided code, I have another py file(plotGauss_Given).
covarMat = np.array([[1.0, 0.0],[0.0, 6.0]]) # 2x2 array

rotation=np.array([[math.cos(-(math.pi/6)), -math.sin(-(math.pi/6))],[math.sin(-(math.pi/6)), math.cos(-(math.pi/6))]])

covarMat2 = np.zeros( (2,2) )
covarMat2= np.concatenate((np.array(rotation.dot(np.array([covarMat[:,0] ]).T)),  
                           np.array(rotation.dot(np.array([covarMat[:,1] ]).T) )), axis=1) 

covarInv = np.linalg.inv(covarMat2)

covarMat3= np.matmul(covarMat2,covarMat)
covarMat3=np.matmul(covarMat3, covarInv)


#checking purpose
[eVal,eVec]=np.linalg.eig(covarMat3)
print(eVal)

covarInv = np.linalg.inv(covarMat3)

mean=np.array([[1.0, 2.0]]).T


def gaussValue(x) : # x is 2-element column vec
    return np.exp(-0.5*np.dot(np.dot(x.T,
                                     covarInv),
                              x))

    
def gaussValue_nonZeroMean(x) : # x is 2-element column vec
    return np.exp(-0.5*np.dot(np.dot((x-mean).T,
                                     covarInv),(x-mean)))    
    

pltRange = np.linspace(-7,7,num=70,endpoint=True)

z1 = np.zeros( (70,70) )

#Uncomment any section individually.

#Section1: Uncomment 58 to 87 for answer 3,4,5

'''

i = 0
for x in pltRange:
    j = 0
    for y in pltRange:
        pt1 = np.array([[x], [y]]) # column vector. 2x1 array.
        z1[j,i] = gaussValue(pt1)
        j = j + 1
    i = i + 1
plt.figure()
CS = plt.contour(pltRange, pltRange, z1)
plt.grid()
plt.savefig('rotated.pdf')


i = 0
for x in pltRange:
    j = 0
    for y in pltRange:
        pt1 = np.array([[x], [y]]) # column vector. 2x1 array.
        z1[j,i] = gaussValue_nonZeroMean(pt1)
        j = j + 1
    i = i + 1
plt.figure()
CS = plt.contour(pltRange, pltRange, z1)
plt.grid()
plt.savefig('rotated_with_center1_2.pdf')

'''



#Section2: Uncomment 93 to 119 for answer 6,7

'''

data_points=np.random.multivariate_normal(mean=np.array([1.0, 2.0]), cov=covarMat3, size=100)
x,y=data_points.T
plt.plot(x,y, 'bo')

plt.savefig('data.pdf')
plt.show()

i = 0
for x in pltRange:
    j = 0
    for y in pltRange:
        pt1 = np.array([[x], [y]]) # column vector. 2x1 array.
        z1[j,i] = gaussValue_nonZeroMean(pt1)
        j = j + 1
    i = i + 1
   
#z1 = np.flipud(z1)
    
plt.figure()
CS = plt.contour(pltRange, pltRange, z1)
plt.grid()
x,y=data_points.T
plt.plot(x,y, 'bo')
plt.savefig('rotated_with_center1_2_with the data.pdf')
'''