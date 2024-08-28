import cv2

img_left = cv2.imread('Explorer_HD720_SN3299_12-02-24_left_half.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('Explorer_HD720_SN3299_12-02-24_right_half.png', cv2.IMREAD_GRAYSCALE)
num_disparities = 16 * 5  # Must be divisible by 16
block_size = 15  # Must be an odd number

sbm = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

disparity = sbm.compute(img_left, img_right)

norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('Disparity Map', norm_disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()

