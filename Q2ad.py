import cv2
import numpy
import glob


def stats(image):
    image_float = numpy.float32(image)
    avg = numpy.mean(image_float, axis=(0, 1))
    stdDev = numpy.std(image_float, axis=(0, 1))
    variance = numpy.var(image_float, axis=(0, 1))
    
    return avg, stdDev, variance


images = []
img = glob.glob('./split_images/*.png')
img = img[:25]


for imPath in img:
    image = cv2.imread(imPath)
    images.append(image)


images = numpy.array(images)


avgs = []
stdDevList = []
variances = []

for image in images:
    avg, stdDev, variance = stats(image)
    avgs.append(avg)
    stdDevList.append(stdDev)
    variances.append(variance)


avgs = numpy.array(avgs)
desvios_padrao = numpy.array(stdDevList)
variances = numpy.array(variances)


avg_total = numpy.mean(avgs, axis=0)
stdDev_total = numpy.mean(desvios_padrao, axis=0)
totalVar = numpy.mean(variances, axis=0)

print(avg_total)
print(stdDev_total)
print(totalVar)



