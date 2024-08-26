from PIL import Image
import os
import random
import matplotlib.pyplot



i = 0
while i < 5:
    if i == 0:
        image = Image.open('./split_images/Explorer_HD720_SN3299_12-02-24_right_half.png')
    if i == 1:
        image = Image.open('./split_images/Explorer_HD720_SN3299_12-04-39_right_half.png')
    if i == 2:
        image = Image.open('./split_images/Explorer_HD720_SN3299_12-23-01_right_half.png')
    if i == 3:
        image = Image.open('./split_images/Explorer_HD720_SN3299_12-30-27_right_half.png')
    if i == 4:
        image = Image.open('./split_images/Explorer_HD720_SN3299_12-33-23_right_half.png')
    
    w, h = image.size
    mode = image.mode
    extrema = image.getextrema()

    print("File number: ")
    print(i)
    print("Dimensions:", w, "x", h)
    print("Color Mode:", mode)
    print("Minimum Gray:")
    for j, color in enumerate(mode):
        print(f"Channel {color}: {extrema[j][0]}")
    print("Maximum Gray:")
    for j, color in enumerate(mode):
        print(f"Channel {color}: {extrema[j][1]}")

    image.close()
    
    i = i + 1

