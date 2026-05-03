# my code here

# 1.
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np

img = image.imread('butterfly.jpg')
gray_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

plt.figure(figsize=(10,6))
plt.imshow(gray_img, cmap='gray')
plt.title(f'Grayscale of Image')
plt.axis('off')
plt.show()

# 2.
average = np.mean(gray_img)
print(f'Mean = {average}')

# 3.
myImageMinusMean = gray_img - average

# 4. 
covariance = np.cov(gray_img, rowvar=False)

# 5. 
eigenValues, eigenVectors = np.linalg.eigh(covariance)

# 6.
idx = np.argsort(eigenValues)[::-1]

eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:, idx]

# 7. 
# I used other values to compare the differences even more
amount_to_keep = [1, 5, 15, 25, 100, 200, 2500]

for amount in amount_to_keep:
    totalVariance = np.sum(eigenValues)
    selectVariance = np.sum(eigenValues[:amount])
    percentage = (selectVariance / totalVariance) * 100
    
    print(f'The top {amount} eigen values are responsible for {percentage} % of the variance.')
    
    # 8.
    eigenValuesToKeep = eigenValues[:amount]
    eigenVectorsToKeep = eigenVectors[:, :amount]
    
    # 9.
    compressedImage = np.matmul(myImageMinusMean, eigenVectorsToKeep)

    # 10.
    lossyUnCompressedImage = np.matmul(compressedImage, np.transpose(eigenVectorsToKeep)) + average

    plt.figure(figsize=(10,6))
    plt.imshow(lossyUnCompressedImage, cmap='gray')
    plt.title(f'PCA Reconstruction with {amount} Components')
    plt.axis('off')
    plt.show()
