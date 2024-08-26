import numpy
import matplotlib
from scipy.fft import fft


# Função retangular
def ret(s, a):
  ans = numpy.where(numpy.logical_and(s >= -a/2, s < a/2), 1, 0)
  return ans



# Função triangular
def tri(s, a):
  ans = numpy.where(numpy.logical_and(s >= -a/2, s < 0), s/a + 0.5, numpy.where(
      numpy.logical_and(s >= 0, s < a/2), -s/a + 0.5, 0
  ))
  return ans



# Função pente
def pen(s, T):
  ans = numpy.where(numpy.abs(s) % T < 1e-10, 1, 0)
  return ans



# Função gaussiana
def gauss(s, sig):
  ans = numpy.exp(-s**2 / (2*sig**2)) / (numpy.sqrt(2*numpy.pi) * sig)
  return ans



# Função impulso
def imp(s):
  ans = numpy.where(s==0, numpy.inf, 0)
  return ans



# Função de derivação
def deriv(func, s):
  ans = s[1] - s[0]
  return numpy.gradient(func, ans)



# Escala
def scale(s, a):
  ans = s * a
  return ans



# Combinação linear
def lin_combination(a, s, functions, coefficients):
  ans = numpy.zeros_like(s)

  for fun, cof in zip(functions, coefficients):
    if fun in [ret, tri, gauss, pen]:
      ans = ans + cof * fun(s, a)

    elif fun is imp:
      ans = ans + cof * fun(s)

    elif fun is deriv:
      ans = ans + cof * fun(numpy.sin(s), s)

    elif fun in [lambda s: numpy.sin(s), lambda s: numpy.cos(s)]:
      ans = ans + cof * fun(s)

    else:
      ans = ans + cof * fun(s)

  return ans



s = numpy.linspace(-8, 8, 1000)
a = 4
sig = 1.0
T = 2

clFun = [ret, tri, gauss, imp, pen, lambda s:deriv(numpy.sin(s), s),
         lambda s: numpy.sin(s), lambda s: numpy.cos(s)]
cfs = [1, 1, 1, 1, 1, 1, 1, 1]

#Plots
matplotlib.pyplot.figure(figsize = (16, 20))

# Plot da função retangular
matplotlib.pyplot.subplot(8, 2, 1)
matplotlib.pyplot.plot(s, ret(s, a))
matplotlib.pyplot.title("Retangular")
matplotlib.pyplot.xlabel("s")
matplotlib.pyplot.ylabel("F(s)")

matplotlib.pyplot.subplot(8, 2, 2)
matplotlib.pyplot.plot(s, numpy.abs(fft(ret(s,a))))
matplotlib.pyplot.title("Fourier de Retangular")
matplotlib.pyplot.xlabel("Frequência")
matplotlib.pyplot.ylabel("Magnitude")

# Plot da função triangular
matplotlib.pyplot.subplot(8, 2, 3)
matplotlib.pyplot.plot(s, ret(s, a))
matplotlib.pyplot.title("Triangular")
matplotlib.pyplot.xlabel("s")
matplotlib.pyplot.ylabel("F(s)")

matplotlib.pyplot.subplot(8, 2, 4)
matplotlib.pyplot.plot(s, numpy.abs(fft(tri(s,a))))
matplotlib.pyplot.title("Fourier de Triangular")
matplotlib.pyplot.xlabel("Frequência")
matplotlib.pyplot.ylabel("Magnitude")

# Plot da função gaussiana
matplotlib.pyplot.subplot(8, 2, 5)
matplotlib.pyplot.plot(s, gauss(s, sig))
matplotlib.pyplot.title("Gaussiana")
matplotlib.pyplot.xlabel("s")
matplotlib.pyplot.ylabel("F(s)")

matplotlib.pyplot.subplot(8, 2, 6)
matplotlib.pyplot.plot(s, numpy.abs(fft(gauss(s,sig))))
matplotlib.pyplot.title("Fourier de Gaussiana")
matplotlib.pyplot.xlabel("Frequência")
matplotlib.pyplot.ylabel("Magnitude")

# Plot da função impulso
matplotlib.pyplot.subplot(8, 2, 7)
matplotlib.pyplot.plot(s, imp(s))
matplotlib.pyplot.title("Impulso")
matplotlib.pyplot.xlabel("s")
matplotlib.pyplot.ylabel("F(s)")

matplotlib.pyplot.subplot(8, 2, 8)
matplotlib.pyplot.plot(s, numpy.abs(fft(imp(s))))
matplotlib.pyplot.title("Fourier de Impulso")
matplotlib.pyplot.xlabel("Frequência")
matplotlib.pyplot.ylabel("Magnitude")

# Plot da função pente
matplotlib.pyplot.subplot(8, 2, 9)
matplotlib.pyplot.plot(s, pen(s, T))
matplotlib.pyplot.title("Pente")
matplotlib.pyplot.xlabel("s")
matplotlib.pyplot.ylabel("F(s)")

matplotlib.pyplot.subplot(8, 2, 10)
matplotlib.pyplot.plot(s, numpy.abs(fft(pen(s, T))))
matplotlib.pyplot.title("Fourier de Pente")
matplotlib.pyplot.xlabel("Frequência")
matplotlib.pyplot.ylabel("Magnitude")

# Plot da função derivada
matplotlib.pyplot.subplot(8, 2, 11)
matplotlib.pyplot.plot(s, deriv(numpy.sin(s), s))
matplotlib.pyplot.title("Derivada")
matplotlib.pyplot.xlabel("s")
matplotlib.pyplot.ylabel("F(s)")

matplotlib.pyplot.subplot(8, 2, 12)
matplotlib.pyplot.plot(s, numpy.abs(fft(deriv(numpy.sin(s), s))))
matplotlib.pyplot.title("Fourier de Derivada")
matplotlib.pyplot.xlabel("Frequência")
matplotlib.pyplot.ylabel("Magnitude")

# Plot da função combinação
matplotlib.pyplot.subplot(8, 2, 13)
matplotlib.pyplot.plot(s, lin_combination(a, s, clFun[:6], cfs[:6]))
matplotlib.pyplot.title("Combinação")
matplotlib.pyplot.xlabel("s")
matplotlib.pyplot.ylabel("F(s)")

matplotlib.pyplot.subplot(8, 2, 14)
matplotlib.pyplot.plot(s, numpy.abs(fft(lin_combination(a, s, clFun[:6], cfs[:6]))))
matplotlib.pyplot.title("Fourier de Combinação")
matplotlib.pyplot.xlabel("Frequência")
matplotlib.pyplot.ylabel("Magnitude")

# Plot da função escala
matplotlib.pyplot.subplot(8, 2, 15)
matplotlib.pyplot.plot(s, scale(s, 2))
matplotlib.pyplot.title("Escala")
matplotlib.pyplot.xlabel("s")
matplotlib.pyplot.ylabel("F(s)")

matplotlib.pyplot.subplot(8, 2, 16)
matplotlib.pyplot.plot(s, numpy.abs(fft(scale(s, 2))))
matplotlib.pyplot.title("Fourier de escala")
matplotlib.pyplot.xlabel("Frequência")
matplotlib.pyplot.ylabel("Magnitude")