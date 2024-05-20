import cv2
import numpy as np
import math
import pywt


img = cv2.imread('Test_Images/2.jpg')


def gamma_enhancement(img):
    # convert img to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow('Normal_RGB Image', img)
    cv2.imshow('HSV image', hsv_img)

    # components splitting from HSV image(hue,saturation,value)
    hue, sat, val = cv2.split(hsv_img)

    # Taking only the VALUE component
    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)
    print(gamma)

    # do gamma correction on VALUE channel
    val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

    # combine new VALUE channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    # hsv image convert to RGB format
    global img_gamma_enhance
    img_gamma_enhance = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    # show results
    cv2.imshow('Gamma Enhanced Image', img_gamma_enhance)

    # save results
    cv2.imwrite(
        'Project_Experiments/Gamma_Enhanced_Image/gamma_enhanced_2.jpg', img_gamma_enhance)


gamma_enhancement(img)  # function call


def sharp_and_histogram_equalization(img):

    # SHARPENING
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    unsharp_masking_img = cv2.addWeighted(
        img, 2.0, gaussian, -1.0, 0)  # (src1,alpha,src2,beta,gamma)

    he = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    img_b = he.apply(img[:, :, 0])
    img_g = he.apply(img[:, :, 1])
    img_r = he.apply(img[:, :, 2])

    global equalized_img
    equalized_img = np.stack((img_b, img_g, img_r), axis=2)

    # show results
    cv2.imshow('Normal_RGB Image', img)
    cv2.imshow("Sharpened Image ", unsharp_masking_img)
    cv2.imshow('Histogram_equalized_img', equalized_img)
    # save results
    cv2.imwrite(
        'Project_Experiments/Sharpened_HE_image/Sharpened_HE_image_2.jpg', equalized_img)


sharp_and_histogram_equalization(img)


def channelTransform(ch1, ch2, shape):
    cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
    cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
    cA1, (cH1, cV1, cD1) = cooef1
    cA2, (cH2, cV2, cD2) = cooef2

    cA = (cA1+cA2)/2
    cH = (cH1 + cH2)/2
    cV = (cV1+cV2)/2
    cD = (cD1+cD2)/2
    fincoC = cA, (cH, cV, cD)
    outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
    outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    return outImageC


def fusion(img1, img2):
    # Seperating channels
    iR1 = img1.copy()
    iR1[:, :, 1] = iR1[:, :, 2] = 0
    iR2 = img2.copy()
    iR2[:, :, 1] = iR2[:, :, 2] = 0

    iG1 = img1.copy()
    iG1[:, :, 0] = iG1[:, :, 2] = 0
    iG2 = img2.copy()
    iG2[:, :, 0] = iG2[:, :, 2] = 0

    iB1 = img1.copy()
    iB1[:, :, 0] = iB1[:, :, 1] = 0
    iB2 = img2.copy()
    iB2[:, :, 0] = iB2[:, :, 1] = 0

    shape = (img1.shape[1], img1.shape[0])
    # Wavelet transformation on red channel
    outImageR = channelTransform(iR1, iR2, shape)
    outImageG = channelTransform(iG1, iG2, shape)
    outImageB = channelTransform(iB1, iB2, shape)

    outImage = img1.copy()
    outImage[:, :, 0] = outImage[:, :, 1] = outImage[:, :, 2] = 0
    outImage[:, :, 0] = outImageR[:, :, 0]
    outImage[:, :, 1] = outImageG[:, :, 1]
    outImage[:, :, 2] = outImageB[:, :, 2]

    outImage = np.multiply(np.divide(
        outImage - np.min(outImage), (np.max(outImage) - np.min(outImage))), 255)
    outImage = outImage.astype(np.uint8)

    cv2.imshow('Fused_img', outImage)

    cv2.imwrite('Project_Experiments/Fusion/Fused_image_2.jpg', outImage)


fusion(img_gamma_enhance, equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
