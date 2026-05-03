# By Drake Setera

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np

# Change this to image of your choosing
img = image.imread('butterfly.jpg')


gray_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])


average = np.mean(gray_img)
print(f'Mean = {average}')

myImageMinusMean = gray_img - average

covariance = np.cov(gray_img, rowvar=False)

eigenValues, eigenVectors = np.linalg.eigh(covariance)

idx = np.argsort(eigenValues)[::-1]

eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:, idx]

# I used different values to compare the differences even more
amount_to_keep = [1, 5, 15, 25, 100, 200, 2500]


fig, axes = plt.subplots(2, 4, figsize=(12, 8), layout='constrained')

axes[0, 0].imshow(gray_img, cmap='gray')
axes[0, 0].set_title(f'Original Image')

for i in range(len(amount_to_keep)):
    
    amount = amount_to_keep[i]
    totalVariance = np.sum(eigenValues)
    selectVariance = np.sum(eigenValues[:amount])
    percentage = (selectVariance / totalVariance) * 100
    
    print(f'The top {amount} eigen values are responsible for {percentage} % of the variance.')
    
    eigenValuesToKeep = eigenValues[:amount]
    eigenVectorsToKeep = eigenVectors[:, :amount]
    
    compressedImage = np.matmul(myImageMinusMean, eigenVectorsToKeep)

    lossyUnCompressedImage = np.matmul(compressedImage, np.transpose(eigenVectorsToKeep)) + average

    axes[(i+1) // 4, (i+1) % 4].imshow(lossyUnCompressedImage, cmap='gray')
    axes[(i+1) // 4, (i+1) % 4].set_title(f'PCA Reconstruction with {amount} Components')
plt.show()
