from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import math
import pywt
a = Tk()
a.title("Image Enhancement")
a.geometry("1200x600")


def sharp_and_histogram_equalization(img):

    # SHARPENING
    gaussian = cv2.GaussianBlur(img, (0, 0), 1.0)
    unsharp_masking_img = cv2.addWeighted(
        img, 2.0, gaussian, -1.0, 0)  # (src1,alpha,src2,beta,gamma)

    he = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # img_b = he.apply(img[:, :, 0])
    # img_g = he.apply(img[:, :, 1])
    # img_r = he.apply(img[:, :, 2])
    img_b = he.apply(unsharp_masking_img[:, :, 0])
    img_g = he.apply(unsharp_masking_img[:, :, 1])
    img_r = he.apply(unsharp_masking_img[:, :, 2])

    global equalized_img
    equalized_img = np.stack((img_b, img_g, img_r), axis=2)

    # show results
    # cv2.imshow('Normal_RGB Image', img)
    # cv2.imshow("Sharpened Image ", unsharp_masking_img)
    # cv2.imshow('Histogram_equalized_img', equalized_img)
    # save results
    # cv2.imwrite(
    #     'Project_Experiments/Sharpened_HE_image/Sharpened_HE_image_2.jpg', equalized_img)

    return equalized_img


def gamma_enhancement(img):
    # convert img to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # cv2.imshow('Normal_RGB Image', img)
    # cv2.imshow('HSV image', hsv_img)

    # components splitting from HSV image(hue,saturation,value)
    hue, sat, val = cv2.split(hsv_img)

    # Taking only the VALUE component
    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid*255)
    gamma=gamma/math.log(mean)
    print(gamma)

    # do gamma correction on VALUE channel
    val_gamma = np.power(val, gamma)
    val_gamma=val_gamma.clip(0, 255)
    val_gamma=val_gamma.astype(np.uint8)

    # combine new VALUE channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    # hsv image convert to RGB format
    global img_gamma_enhance
    img_gamma_enhance = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    # show results
    # cv2.imshow('Gamma Enhanced Image', img_gamma_enhance)

    # save results
    # cv2.imwrite(
    #     'Project_Experiments/Gamma_Enhanced_Image/gamma_enhanced_2.jpg', img_gamma_enhance)

    return img_gamma_enhance


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

    # here
    # cv2.imshow('Fused_img', outImage)
    print(outImage)

    outImage=cv2.cvtColor(outImage,cv2.COLOR_BGR2RGB)

    my_final_image = ImageTk.PhotoImage(image=Image.fromarray(outImage))

    out_label.config(image=my_final_image)
    # out_label = Label(f4, image=my_final_image)
    out_label.image = my_final_image
    #out_label.pack(side="left", padx=50, pady=60)
    out_label.pack(anchor=CENTER,pady=180)
    # cv2.imwrite('Project_Experiments/Fusion/Fused_image_2.jpg', outImage)

    return outImage


def enhance():

    list_box.insert(1, "Loading Image")
    list_box.insert(2, "")
    list_box.insert(3, "Gamma Enhancing")
    list_box.insert(4, "")
    list_box.insert(5, "Sharpening")
    list_box.insert(6, "")
    list_box.insert(7, "Histogram Equalizing")
    list_box.insert(8, "")
    list_box.insert(9, "Fusion")

    img = cv2.imread(path)

    gamma_enhancement(img)  # function call
    sharp_and_histogram_equalization(img)
    fusion(img_gamma_enhance, equalized_img)

   
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="Lavender")
    f1.place(x=0, y=0, width=500, height=690)
    f1.config()

    input_label = Label(f1, text="INPUT", font="arial 16", bg="Lavender")
    input_label.place(x=200, y=20)

    upload_pic_button = Button(
        f1, text="Upload Picture", command=Upload, bg="pink")
    upload_pic_button.place(x=190, y=100)

    global label
    label=Label(f1,bg="Lavender")

    f3 = Frame(f, bg="Salmon")
    f3.place(x=500, y=0, width=240, height=690)
    f3.config()

    name_label = Label(f3, text="Process", font="arial 14", bg="Salmon")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()

    enhance_button = Button(
        f3, text="Enhance", command=enhance, bg="deepskyblue")
    enhance_button.place(x=90, y=300)

    global f4
    f4 = Frame(f, bg="light green")
    f4.place(x=740, y=0, width=460, height=690)
    f4.config()

    result_label = Label(f4, text="RESULT", font="arial 16", bg="light green")
    result_label.place(x=180, y=10)

    global out_label
    out_label=Label(f4,bg="light green")


def Upload():

    global path
    label.config(image='')
    out_label.config(image='')
    list_box.delete(0,END)
    path = askopenfilename(title='Open a file',
                           initialdir='D:\\DENNY\\Implementable_OR_not\\LL_img_enh_via_fusion\\Test_Images',
                           filetypes=(("JPG", "*.jpg"), ("PNG", "*.png"), ("JPEG", "*.jpeg")))
    print("<<<<<<<<<<<<<", path)
    image = Image.open(path)
    global imagename
    imagename = ImageTk.PhotoImage(image)# .resize((320, 320))
    # label = Label(f1, image=imagename)
    label.config(image=imagename)
    label.image = imagename
    label.pack(anchor=CENTER,pady=180)


def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="Aquamarine")
    f.pack(side="top", fill="both", expand=True)

    home_label = Label(f, text="Image Enhancement",
                       font="arial 35", bg="Aquamarine")
    home_label.place(x=390, y=250)


f = Frame(a, bg="Aquamarine")
f.pack(side="top", fill="both", expand=True)

home_label = Label(f, text="Image Enhancement",
                   font="arial 35", bg="Aquamarine")
home_label.place(x=390, y=250)

m = Menu(a)
m.add_command(label="Home", command=Home)
checkmenu = Menu(m)
m.add_command(label="Check", command=Check)
a.config(menu=m)


a.mainloop()
