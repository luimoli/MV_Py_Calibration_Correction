import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from utils import smv_colour
from utils import mcc_detect_color_checker
from utils.deltaE.deltaC_2000_np import delta_C_CIE2000
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000
from utils import color_science



def image_correction(image, image_color_space, rgb_gain, ill_gain, ccm, white_balance=True,
                     illumination_gain=True, ccm_correction=True):
    if white_balance:
        image = image * rgb_gain[None, None]
        image = np.clip(image, 0, 1)
    if illumination_gain:
        image = image * ill_gain
        image = np.clip(image, 0, 1)

    if ccm_correction:
        print(ccm.shape[0])
        if ccm.shape[0] == 4:
            image = np.einsum('ic, hwc->hwi', ccm[0:3].T, image) + ccm[3][None, None]
        else:
            image = np.einsum('ic, hwc->hwi', ccm.T, image)
    image = np.clip(image, 0, 1)
    return image


# def predict_ccm(cct_list, ccm_list, cct):
#     if cct < cct_list[0]:
#         return ccm_list[0]
#     if cct > cct_list[1]:
#         return ccm_list[0]
#
#     for i in range(1, len(cct_list)-1):
#         if cct < cct_list[i]:
#             cct1 = cct_list[i - 1]
#             cct2 = cct_list[i]
#             ccm1 = ccm_list[i - 1]
#             ccm2 = ccm_list[i]
#             alpha = (1 / cct - 1 / cct2) / (1 / cct1 - 1 / cct2)
#             ccm = alpha * ccm1 + (1 - alpha) * ccm2
#             print(cct1, cct, cct2)
#             print(ccm1, ccm, ccm1)
#             return ccm


def calculate_colorchecker_value(image, sorted_centroid, length):
    sorted_centroid2 = np.int32(sorted_centroid)
    mean_value = np.empty((sorted_centroid.shape[0], 3))
    for i in range(len(sorted_centroid)):
        mean_value[i] = np.mean(image[sorted_centroid2[i, 1] - length:sorted_centroid2[i, 1] + length,
                                sorted_centroid2[i, 0] - length:sorted_centroid2[i, 0] + length], axis=(0, 1))
    return np.float32(mean_value)


def evaluate_result(image, image_color_space, sorted_centroid, ideal_lab):
    if sorted_centroid is None:
        sorted_centroid, clusters, marker_image = mcc_detect_color_checker.detect_color_checker(image)
    result_cc_mean = calculate_colorchecker_value(image, sorted_centroid, 50)
    result_cc_mean = np.clip(result_cc_mean, 0, 1)
    if image_color_space == "srgb":
        result_cc_mean_lab = smv_colour.XYZ2Lab(smv_colour.RGB2XYZ(color_science.gamma_reverse(result_cc_mean), "bt709"))
    else:
        result_cc_mean_lab = smv_colour.XYZ2Lab(smv_colour.RGB2XYZ(result_cc_mean, "bt709"))
    result_cc_mean_lab = np.array(result_cc_mean_lab)
    deltaC = delta_C_CIE2000(result_cc_mean_lab, ideal_lab)
    # deltaE = colour.delta_E(result_cc_mean_lab, self.ideal_lab, 'CIE 2000')
    deltaE = delta_E_CIE2000(result_cc_mean_lab, ideal_lab)
    return deltaC, deltaE


def draw_gt_in_image(image, image_color_space, deltaE, sorted_centroid, ideal_linear_rgb):
    if sorted_centroid is None:
        sorted_centroid, clusters, marker_image = mcc_detect_color_checker.detect_color_checker(image)
    image_gt = image.copy()
    length = 50
    sorted_centroid = np.int32(sorted_centroid)
    for i in range(len(sorted_centroid)):
        if image_color_space.lower() == "linear":
            image_gt[sorted_centroid[i, 1] - length:sorted_centroid[i, 1] + length,
            sorted_centroid[i, 0] - length:sorted_centroid[i, 0] + length] = ideal_linear_rgb[i]
        else:
            ideal_srgb = color_science.gamma(ideal_linear_rgb)
            image_gt[sorted_centroid[i, 1] - length:sorted_centroid[i, 1] + length,
            sorted_centroid[i, 0] - length:sorted_centroid[i, 0] + length] = ideal_srgb[i]
        cv2.putText(image_gt, str(round(deltaE[i], 1)), np.int32(sorted_centroid[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    return image_gt


def predict_ccm(cct_ccm_dict, cct):
    cct_list = sorted(cct_ccm_dict.keys())
    print(cct_list, cct)
    if cct <= cct_list[0]:
        return cct_ccm_dict[cct_list[0]]
    elif cct >= cct_list[-1]:
        return cct_ccm_dict[cct_list[-1]]
    for i in range(1, len(cct_list)):
        if float(cct) <= cct_list[i]:
            cct_left = cct_list[i-1]
            cct_right = cct_list[i]
            ccm_left = cct_ccm_dict[cct_list[i - 1]]
            ccm_right = cct_ccm_dict[cct_list[i]]
            alpha = (1/cct - 1/cct_right) / (1/cct_left - 1/cct_right)
            ccm = alpha * ccm_left + (1-alpha) * ccm_right
            break

    return ccm

if __name__ == '__main__':
    image = cv2.imread(r"E:/code/color_calibration_and_correction/data/mindvision/exposure30.jpg")
    image = image[..., ::-1] / 255

    cct = color_science.calculate_cct(np.array([1/1.0388633, 1, 1/4.3960304]))
    a = np.load("../test.npy", allow_pickle=True).item()
    ccm = np.array(predict_ccm(a, cct))
    print(ccm, cct)
    image_calib = image_correction(image, "linear", np.array([1.0388633, 1, 4.3960304]), 1, ccm, True, True, True)
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(image_calib)
    plt.show()














