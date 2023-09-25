import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from utils import color_science
from utils import smv_colour
import torch
from utils import color_correction

image_wp = cv2.imread("save_image_1650353183.316039.png", -1) / 65535.
image_ori = cv2.imread("save_image_1650353203.586852.png", -1) / 65535.


# white_point = image / image[..., 1:2]
white_point = image_wp
wb_gain = image_wp[..., 1:2] / image_wp
image_wp_wb = image_wp * wb_gain
image_ori_wb = image_ori * wb_gain

white_point_XYZ = smv_colour.RGB2XYZ(torch.from_numpy(np.float32(white_point)), "bt709")
white_point_xyY = smv_colour.XYZ2xyY(white_point_XYZ)
x, y = white_point_xyY[..., 0], white_point_xyY[..., 1]
n = (x - 0.3320) / (y - 0.1858)
cct = -449 * n ** 3 + 3525 * n ** 2 - 6823.3 * n + 5520.33
cct = np.array(cct)

data = np.load("calibration_3nh.npy", allow_pickle=True).item()
cct1 = 4739.57958984375
cct2 = 6069.05029296875

print(data.keys())

# cct[cct > cct2] = cct2
# cct[cct < cct1] = cct1

ccm1 = data[cct1]
ccm2 = data[cct2]
image_wb_ccm1 = color_correction.image_correction(image_ori_wb, "linear", None, None, ccm1, False, False, True)
image_wb_ccm2 = color_correction.image_correction(image_ori_wb, "linear", None, None, ccm2, False, False, True)

alpha = (1 / cct - 1 / cct2) / (1 / cct1 - 1 / cct2)
alpha = alpha[..., None]

hist = cv2.calcHist([cct[..., None]], [0], None, [100], [2000, 10000])
plt.figure()
plt.imshow(cct)
plt.show()
exit()
result = alpha * image_wb_ccm1 + (1-alpha) * image_wb_ccm2

cv2.imwrite("2.png", np.uint8(result[..., ::-1] ** (1/2.2) * 255))

plt.figure()
plt.imshow(cct)
plt.figure()
plt.imshow(image_ori_wb)
plt.figure()
plt.imshow(result)
plt.show()






