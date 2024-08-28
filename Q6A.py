import numpy as numpy
import matplotlib.pyplot
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter
from skimage import io, color

path = './split_images/Explorer_HD720_SN3299_12-47-32_right_half.png'
imgSrc = io.imread(path)

if len(imgSrc.shape) == 3 and imgSrc.shape[2] == 4:
    imgSrc = imgSrc[:, :, :3]

if len(imgSrc.shape) == 3:
    imgSrc = color.rgb2gray(imgSrc)

imgSrc = (imgSrc - numpy.min(imgSrc)) / (numpy.max(imgSrc) - numpy.min(imgSrc))
imgSrc_smooth = gaussian_filter(imgSrc, sigma=1)
dz_dx = numpy.gradient(imgSrc_smooth, axis=1)
dz_dy = numpy.gradient(imgSrc_smooth, axis=0)

dim = imgSrc.shape
z = numpy.zeros_like(imgSrc_smooth)

for i in range(1, dim[0]):
    z[i, :] = z[i-1, :] + dz_dy[i, :]

for j in range(1, dim[1]):
    z[:, j] = z[:, j-1] + dz_dx[:, j]

X, Y = numpy.meshgrid(numpy.arange(dim[1]), numpy.arange(dim[0]))
fig = go.Figure(data=[go.Surface(z=z, x=X, y=Y, colorscale='Viridis')])
fig.update_layout(title='Estimated Shape (shading)', autosize=False,
                  width=800, height=800, margin=dict(l=65, r=50, b=65, t=90))

matplotlib.pyplot.imshow(imgSrc, cmap='gray')
matplotlib.pyplot.title("Real Img")
matplotlib.pyplot.colorbar()
matplotlib.pyplot.show()

fig.show()