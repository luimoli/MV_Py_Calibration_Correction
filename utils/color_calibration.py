import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import torch

from utils import smv_colour
from utils import color_science
from utils import mcc_detect_color_checker
from utils import color_correction
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, demosaicing_CFA_Bayer_bilinear


ColorChecker2005_3nh = np.array([[36.38,11.99,13.76],
                                [64.68,15.88,17.79],
                                [49,-2.06,-21.12],
                                [42.22,-14.79,22.29],
                                [54.27,11.91,-25.66],
                                [70.66,-32.52,2],
                                [61.2,32.23,55.33],
                                [40.01,17.42,-45.62],
                                [49.42,46.49,15.06],
                                [29.26,25.67,-23.12],
                                [72.7,-27.19,58.51],
                                [70.67,15.17,67.2],
                                [28.77,22.67,-49.29],
                                [53.68,-41.21,33.05],
                                [39.96,49.04,23.62],
                                [80.65,-1.14,80.45],
                                [50.79,51.88,-16.79],
                                [51.99,-24.22,-26.5],
                                [96.23,-0.15,1.56],
                                [80.97,-0.46,-0.57],
                                [66.41,-0.23,-0.39],
                                [49.88,-0.32,-0.01],
                                [34.88,-0.31,-1.17],
                                [18.7,0.22,-1.71,]])


class ColorCalibration():
    def __init__(self):
        self.white_point = np.zeros(3)
        self.wb_gain = np.ones(3)
        self.chart_19_mean = 0.90616
        self.chart_20_mean = 0.58826


    # def calculate_cct(self, white_point):
    #     print(white_point)
    #     white_point_XYZ = smv_colour.RGB2XYZ(torch.from_numpy(np.float32(white_point)), "bt709")
    #     white_point_xyY = smv_colour.XYZ2xyY(white_point_XYZ)
    #     x, y = white_point_xyY[0], white_point_xyY[1]
    #     n = (x - 0.3320) / (y - 0.1858)
    #     cct = -449 * n**3 + 3525 * n**2 - 6823.3 * n + 5520.33
    #     return cct


    def calculate_WBgain(self, charts_linear_rgb):
        max_value = np.max(charts_linear_rgb[18:20], axis=-1, keepdims=True)
        wb_gain = np.mean(max_value / charts_linear_rgb[18:20], axis=0)
        return wb_gain


    def calculate_Ygain(self, wb_charts_linear_rgb):
        mean_value = np.mean(wb_charts_linear_rgb[18:20], axis=-1)   #2
        print("mean_value:", mean_value)
        print("wb_charts_linear_rgb[18:20]:", wb_charts_linear_rgb[18:20])
        return 0.5 * (self.chart_19_mean / mean_value[0]) + 0.5 * (self.chart_20_mean / mean_value[1])


    def calculate_ccm(self, charts_rgb, ccm_type, cc_value):
        """
        charts_image: 24*3 rgb format float64
        """
        cc_value = np.float64(cc_value[:, None, :])
        cc_model = cv2.ccm_ColorCorrectionModel(charts_rgb, cc_value, cv2.ccm.COLOR_SPACE_LAB_D65_2)

        # cc_model = cv2.ccm_ColorCorrectionModel(charts_rgb, cv2.ccm.COLORCHECKER_Macbeth)

        cc_model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        if ccm_type == 0:
            cc_model.setCCM_TYPE(cv2.ccm.CCM_3X3)
        else:
            cc_model.setCCM_TYPE(cv2.ccm.CCM_4X3)

        cc_model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        cc_model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        cc_model.setLinearGamma(1)
        cc_model.setLinearDegree(3)
        # weight = np.ones(24)
        # weight[14] = 2
        # weight[18] = 20
        # weight[19:] = 0
        # cc_model.setWeightsList(weight)
        cc_model.run()
        ccm = cc_model.getCCM()
        loss = cc_model.getLoss()
        print("loss:", loss)
        return ccm


def read_raw_data(data, height, width):
    data = data - 512
    data = data / 16
    data = np.where(data <= 30, 1.97130908 * data - 29.56680552, data) + 30
    data /= 4096

    # data = (data) / 65535
    return data


if __name__ == '__main__':
    ideal_lab = np.float32(np.loadtxt("E:/code/Color_Calibration_Correction/data/real_lab_imatest.csv", delimiter=','))
    # image = cv2.imread(r"E:/code/color_calibration_and_correction/data/mindvision/d65_colorchecker.jpg")
    # image = cv2.imread(r"E:/code/color_calibration_and_correction/data/mindvision/exposure30.jpg")
    # data = np.fromfile(r"C:\Users\30880\Desktop\mindvision\image.RAW", dtype=np.uint16).reshape((2048, 2448))
    # data = read_raw_data(data, 2048, 2448)
    # data = demosaicing_CFA_Bayer_Menon2007(data, pattern="RGGB")[..., ::-1]
    # data = np.float32(data)
    # image = data

    # data = cv2.imread(r"C:\Users\30880\Desktop\mindvision\mindvision\SM-SUA501GC-T-Snapshot-20220407-155513-744-6178296820.PNG") / 255.

    # data = np.float32(data[..., ::-1])
    data = cv2.imread(r"E:/code/color_calibration_and_correction/data/mindvision/exposure30.jpg")
    data2 = np.copy(data)
    data = np.float32(data[..., ::-1] / 255.)

    sorted_centroid, charts_RGB, marker_image = mcc_detect_color_checker.detect_color_checker(data2)
    ideal_linear_rgb = smv_colour.XYZ2RGB(smv_colour.Lab2XYZ(torch.from_numpy(ideal_lab)), 'bt709').numpy()
    color_calibration = ColorCalibration()
    wb_gain = color_calibration.calculate_WBgain(charts_RGB)
    print(wb_gain)
    Y_gain = color_calibration.calculate_Ygain(charts_RGB * wb_gain[None])
    wb_charts_RGB = (charts_RGB * wb_gain[None] * Y_gain)[:, None]
    ccm = color_calibration.calculate_ccm(np.float64(wb_charts_RGB), 0, ideal_lab)

    exit()

    sorted_centroid, charts_RGB, marker_image = mcc_detect_color_checker.detect_color_checker(dataxxxx)
    ideal_linear_rgb = smv_colour.XYZ2RGB(smv_colour.Lab2XYZ(torch.from_numpy(ideal_lab)), 'bt709').numpy()
    color_calibration = ColorCalibration()
    wb_gain = color_calibration.calculate_WBgain(charts_RGB)
    Y_gain = color_calibration.calculate_Ygain(charts_RGB * wb_gain[None])
    wb_charts_RGB = (charts_RGB * wb_gain[None] * Y_gain)[:, None]
    image_calib = color_correction.image_correction(dataxxxx, "linear", wb_gain, Y_gain, ccm, True, True, True)
    deltaC, deltaE = color_correction.evaluate_result(image_calib, "linear", sorted_centroid, ideal_lab)
    image_gt = color_correction.draw_gt_in_image(image_calib, "linear", deltaC, sorted_centroid, ideal_linear_rgb)
    # cv2.imwite("image_gt.png", np.uint8(image_gt ** (1 / 2.2) * 255))
    print(deltaC.mean(), deltaE.mean())

    plt.figure()
    plt.imshow(image_gt ** (1 / 2.2))
    plt.show()