import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt



def transform_points_forward(_T, X):
    p = np.array([[X[0]], [X[1]], [1]])
    # xt = _T * p
    xt = np.dot(_T, p)
    return np.array([xt[0, 0]/xt[2, 0], xt[1,0]/xt[2,0]])

def CGetCheckerCentroid(checker):
    cellchart = np.array([[1.50, 1.50], [4.25, 1.50], [7.00, 1.50], [9.75, 1.50], [12.50, 1.50], [15.25, 1.50],
    [1.50, 4.25], [4.25, 4.25], [7.00, 4.25], [9.75, 4.25], [12.50, 4.25], [15.25, 4.25], [1.50, 7.00],
    [4.25, 7.00], [7.00, 7.00], [9.75, 7.00], [12.50, 7.00], [15.25, 7.00], [1.50, 9.75], [4.25, 9.75],
    [7.00, 9.75], [9.75, 9.75], [12.50, 9.75], [15.25, 9.75]])
    center = checker.getCenter()
    box = checker.getBox()
    size = np.array([4, 6])
    boxsize = np.array([11.25, 16.75])
    fbox = np.array([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])
    pixel_distance = ((box[0] - box[1]) ** 2).sum() ** 0.5
    block_pixel = pixel_distance / 7
    ccT = cv2.getPerspectiveTransform(np.float32(fbox), np.float32(box))
    sorted_centroid = []
    for i in range(24):
         Xt = transform_points_forward(ccT, cellchart[i])
         sorted_centroid.append(Xt)
    return np.array(sorted_centroid), block_pixel

# def detect_color_checker(image):
#     detector = cv2.mcc.CCheckerDetector_create()
#     detector.process(np.uint8(255 * image), cv2.mcc.MCC24, 1, True)
#
#     checkers = detector.getListColorChecker()
#     print(len(checkers))
#     checker = checkers[0]
#     cdraw = cv2.mcc.CCheckerDraw_create(checker)
#     img_draw = image.copy()
#     cdraw.draw(img_draw)
#
#     chartsRGB = checker.getChartsRGB()
#     width, height = chartsRGB.shape[:2]
#     roi = chartsRGB[0:width, 1]
#     rows = int(roi.shape[:1][0])
#     charts_RGB = chartsRGB[:, 1].copy().reshape(int(rows / 3), 1, 3)
#
#     sorted_centroid = CGetCheckerCentroid(checker)
#     marker_image = np.copy(image)
#     print(sorted_centroid)
#     for num, centroid in enumerate(sorted_centroid):
#         cv2.putText(marker_image, str(num), np.int32(centroid), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
#     return np.array(sorted_centroid), charts_RGB, marker_image


def calculate_colorchecker_value(image, sorted_centroid, length):
    sorted_centroid2 = np.int32(sorted_centroid)
    length = int(length)
    mean_value = np.empty((sorted_centroid.shape[0], 3))
    for i in range(len(sorted_centroid)):
        mean_value[i] = np.mean(image[sorted_centroid2[i, 1] - length:sorted_centroid2[i, 1] + length,
                                sorted_centroid2[i, 0] - length:sorted_centroid2[i, 0] + length], axis=(0, 1))
    return np.float32(mean_value)

def detect_color_checker(image):
    # image: rgb 0-1
    image_uint8 = np.uint8(255 * image[..., ::-1])
    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(image_uint8, cv2.mcc.MCC24, 1, True)

    checkers = detector.getListColorChecker()
    checker = checkers[0]
    cdraw = cv2.mcc.CCheckerDraw_create(checker)
    img_draw = image_uint8.copy()
    cdraw.draw(img_draw)

    sorted_centroid, block_pixel = CGetCheckerCentroid(checker)
    marker_image = np.copy(image_uint8)
    charts_RGB = calculate_colorchecker_value(image, np.int32(sorted_centroid), block_pixel//4)

    for num, centroid in enumerate(sorted_centroid):
        cv2.putText(marker_image, str(num), np.int32(centroid), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
    return np.array(sorted_centroid), charts_RGB, marker_image



if __name__ == '__main__':
    image = cv2.imread(r"E:\code\color_calibration_and_correction\data\mindvision\exposure30.jpg")
    _, _, image = detect_color_checker(image)
    plt.figure()
    plt.imshow(image)
    plt.show()

