import cv2
import numpy
import matplotlib.pyplot
import glob

def covar(src):
    srcGrayscale = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    cx, cy, _ = srcGrayscale.shape
    cx = cx / 2
    cy = cy / 2
    
    minX = cx - 16
    maxX = cx + 16
    minY = cy - 16
    maxY = cy + 16
    
    area = srcGrayscale[minY : maxY, minX : maxX]
    cov = numpy.cov(area)
    return cov


covars = []
images = []

image = glob.glob('./split_images/*.png')
image = image[:25]

for path in image:
    image = cv2.imread(path)
    images.append(image)

for image in images:
    cov = covar(image)
    covars.append(cov)

avg = numpy.mean(covars, axis=0)
matplotlib.pyplot.imshow(avg, cmap='viridis', interpolation='nearest')
matplotlib.pyplot.title('Average Covariance')
matplotlib.pyplot.colorbar()
matplotlib.pyplot.show()