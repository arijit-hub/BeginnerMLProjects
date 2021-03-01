# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 02:35:19 2021

@author: Arijit
"""

import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt

original_image = cv2.imread('car.png')

flattened_img = original_image.reshape(-1 , 3)

gmm = GMM(n_components = 3 , covariance_type = 'full').fit(flattened_img)

segmented_img = gmm.predict(flattened_img)

final_img = segmented_img.reshape(original_image.shape[0] , original_image.shape[1])


plt.imshow(final_img)

plt.savefig('segmented_image.jpg')