import matplotlib.pyplot
import numpy
from PIL import Image

def plot_pixelLine(imagePath, lineNumber):
    image = Image.open(imagePath)
    width, height = image.size
    
    if lineNumber >= height:
        print("Line does not exist")
        return 0


    imageArray = numpy.array(image)
    pixelLine = imageArray[lineNumber, :, :]


    meanLine = numpy.mean(pixelLine, axis=1)
    stdLine = numpy.std(pixelLine, axis=1)

    upper_curve = meanLine + stdLine
    lower_curve = meanLine - stdLine

    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(meanLine, label='Avg', color='red')
    matplotlib.pyplot.fill_between(range(width), lower_curve, upper_curve, alpha=0.3, color='black', label='Avg ± Standard Deviation')
    matplotlib.pyplot.fill_between(range(width), meanLine, upper_curve, alpha=0.3, color='purple', label='Avg + Standard Deviation')
    matplotlib.pyplot.xlabel('Pixel')
    matplotlib.pyplot.ylabel('Value')
    matplotlib.pyplot.title(f'Curves to line {lineNumber}')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

imagePath = './split_images/Explorer_HD720_SN3299_12-02-24_right_half.png'
lineNumber = 80

plot_pixelLine(imagePath, lineNumber)