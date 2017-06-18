import numpy as np
import cv2
from skimage import data

image_src = cv2.imread('./results/result1.tif')
# image_src = data.imread('D:/piro/piro3/results/result1.tif', True)

# gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(image_src, 250, 255,0)

image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros((600,600), np.uint8)
largest_areas = sorted(contours, key=cv2.contourArea)
cv2.drawContours(mask, largest_areas, 0, (255,255), -1)
removed = cv2.add(image_src, mask)

cv2.imwrite("removed.png", removed)

# import os
# from skimage import data, io
# from skimage.morphology import erosion, dilation, opening, closing, white_tophat, opening
# from PIL import Image
# from scipy.signal import medfilt2d
# from skimage import morphology
# import numpy as np
# import cv2
# from scipy import ndimage
# from skimage import data, io, filters, morphology, feature, exposure, measure
#
#
# def get_binary_image(image):
#     ret, thresh = cv2.threshold(image, 127, 255, 0)
#     kernel = np.ones((4, 4), np.uint8)
#     try:
#         closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     except:
#         closed_thresh = thresh
#     return closed_thresh
#
#
# def get_images_names_list(images_path, labels_path):
#     names_list = os.listdir(images_path)
#     full_path_names = [[os.path.splitext(x)[0], images_path + x, labels_path + os.path.splitext(x)[0] + '.tif'] for x in
#                        names_list]
#     return full_path_names
#
#
# def draw_only_long_roads(image):
#     mask = np.zeros(image.shape, np.uint8)
#     ret, thresh = cv2.threshold(image, 127, 255, 0)
#     kernel = np.ones((4, 4), np.uint8)
#     # closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     contours, hierarchy = cv2.findContours(ret, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 100:
#             cv2.drawContours(mask, [cnt], 0, 255, -1)
#     return mask
#
#
# def get_image(path):
#     image = data.imread(path, False)
#     return image
#
# def getContours(blackAndWhiteImage,limitContourLength):
#     contours = measure.find_contours(blackAndWhiteImage,0.9)
#     contours = [contour for contour in contours if len(contour)>limitContourLength]
#     return contours
#
#
# def get_binary_image(image):
#     riffle = 127.5
#     height, width = np.shape(image)
#     im = np.zeros((height, width))
#     for i in range(height):
#         for j in range(width):
#             if image[i, j] > riffle:
#                 im[i, j] = 1
#             else:
#                 im[i, j] = 0
#     return im
#
#
# def show_image(image):
#     im = Image.fromarray(image)
#     im.show()
#
#
# def main():
#     filepath = './results/'
#     data = get_images_names_list(filepath, filepath)
#     image = get_image(data[0][1])
#     im = Image.fromarray(image)
#     im_gray = im.convert('L')
#     #contours, hierarchy = cv2.findContours(np.array(im_gray), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     #mask = np.zeros(image.shape, np.uint8)
#     #for cnt in contours:
#         # if cv2.contourArea(cnt) > 100:
#         #     cv2.drawContours(mask, [cnt], 0, 255, -1)
#
#     contours = measure.find_contours(im_gray, 0.9)
#     contours = [contour for contour in contours if len(contour) > 100]
#     np_contours = np.array(contours)
#     print(len(contours))
#     cnt = cv2.findContours(np.array(im_gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
#
#     mask = np.zeros(image.shape[:2])
#     cv2.drawContours(mask, [cnt], 0, 255, 0)
#
#     draw_image = Image.fromarray(mask)
#     draw_image.show()
#     #
#     # im_gray2 = Image.fromarray(mask)
#     # im_gray2.show()
#     #
#     # print('sdgdsg')
#     # show_image(image)
#     # i = Image.fromarray(image)
#     # PILImage = Image.fromarray(image)
#     # bw_image = PILImage.convert('1')
#     # show_image(bw_image)
#     # #image_file = Image.open("convert_image.png")  # open colour image
#     # #image_file = image_file.convert('1')
#     # show_image(image)
#     # # image = get_binary_image(image)
#     # super_image = draw_only_long_roads(image)
#     # show_image(super_image)
#     # cleaned = morphology.remove_small_objects(image, min_size=0, connectivity=2)
#     # # opening_image = medfilt2d(image, (5,5))
#     # show_image(image)
#     # show_image(np.array(cleaned, int))
#     #
#
# if __name__ == '__main__':
#     main()
