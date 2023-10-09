import numpy as np
import cv2


def low_pass(img):
    if isinstance(img, str):
        img = cv2.imread(img)

    # do dft saving as complex output
    dft = np.fft.fft2(img, axes=(0,1))

    # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)

    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20

    # create circle mask
    radius = 32
    mask = np.zeros_like(img)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

    # blur the mask
    mask2 = cv2.GaussianBlur(mask, (19,19), 0)

    # apply mask to dft_shift
    dft_shift_masked = np.multiply(dft_shift,mask) / 255
    dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255


    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(dft_shift)
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


    # do idft saving as complex output
    img_back = np.fft.ifft2(back_ishift, axes=(0,1))
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

    # combine complex real and imaginary components to form (the magnitude for) the original image again
    img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)
    img_filtered2 = np.abs(img_filtered2).clip(0,255).astype(np.uint8)


    cv2.imshow("ORIGINAL", img)
    cv2.imshow("SPECTRUM", spec)
    cv2.imshow("MASK", mask)
    cv2.imshow("MASK2", mask2)
    cv2.imshow("ORIGINAL DFT/IFT ROUND TRIP", img_back)
    cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
    cv2.imshow("FILTERED2 DFT/IFT ROUND TRIP", img_filtered2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # write result to disk
    cv2.imwrite("dft_numpy_mask.png", mask)
    cv2.imwrite("dft_numpy_mask_blurred.png", mask2)
    cv2.imwrite("dft_numpy_roundtrip.png", img_back)
    cv2.imwrite("dft_numpy_lowpass_filtered1.png", img_filtered)
    cv2.imwrite("dft_numpy_lowpass_filtered2.png", img_filtered2)


def high_pass_edge(img):
    if isinstance(img, str):
        img = cv2.imread(img)

    # do dft saving as complex output
    dft = np.fft.fft2(img, axes=(0,1))

    # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)

    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20

    # create white circle mask on black background and invert so black circle on white background
    radius = 32
    mask = np.zeros_like(img)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
    mask = 255 - mask

    # blur the mask
    mask2 = cv2.GaussianBlur(mask, (19,19), 0)

    # apply mask to dft_shift
    dft_shift_masked = np.multiply(dft_shift,mask) / 255
    dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255


    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(dft_shift)
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


    # do idft saving as complex output
    img_back = np.fft.ifft2(back_ishift, axes=(0,1))
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

    # combine complex real and imaginary components to form (the magnitude for) the original image again
    # multiply by 3 to increase brightness
    img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
    img_filtered = np.abs(3*img_filtered).clip(0,255).astype(np.uint8)
    img_filtered2 = np.abs(3*img_filtered2).clip(0,255).astype(np.uint8)


    cv2.imshow("ORIGINAL", img)
    cv2.imshow("SPECTRUM", spec)
    cv2.imshow("MASK", mask)
    cv2.imshow("MASK2", mask2)
    cv2.imshow("ORIGINAL DFT/IFT ROUND TRIP", img_back)
    cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
    cv2.imshow("FILTERED2 DFT/IFT ROUND TRIP", img_filtered2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # write result to disk
    cv2.imwrite("dft_numpy_mask_highpass.png", mask)
    cv2.imwrite("dft_numpy_mask_highpass_blurred.png", mask2)
    cv2.imwrite("dft_numpy_roundtrip.png", img_back)
    cv2.imwrite("dft_numpy_highpass_filtered1.png", img_filtered)
    cv2.imwrite("dft_numpy_highpass_filtered2.png", img_filtered2)


def high_boost(img):
    if isinstance(img, str):
        img = cv2.imread(img)

    # do dft saving as complex output
    dft = np.fft.fft2(img, axes=(0,1))

    # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)

    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20

    # create white circle mask on black background and invert so black circle on white background
    # as highpass filter
    radius = 32
    mask = np.zeros_like(img, dtype=np.float32)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx,cy), radius, (1,1,1), -1)[0]
    mask = 1 - mask

    # high boost filter (sharpening) = 1 + fraction of high pass filter
    mask = 1 + 0.5*mask 

    # blur the mask
    mask2 = cv2.GaussianBlur(mask, (19,19), 0)

    # apply mask to dft_shift
    dft_shift_masked = np.multiply(dft_shift,mask)
    dft_shift_masked2 = np.multiply(dft_shift,mask2)

    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(dft_shift)
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)

    # do idft saving as complex output
    img_back = np.fft.ifft2(back_ishift, axes=(0,1))
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

    # combine complex real and imaginary components to form (the magnitude for) the original image again
    img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)
    img_filtered2 = np.abs(img_filtered2).clip(0,255).astype(np.uint8)

    cv2.imshow("ORIGINAL", img)
    cv2.imshow("SPECTRUM", spec)
    cv2.imshow("MASK", mask)
    cv2.imshow("MASK2", mask2)
    cv2.imshow("ORIGINAL DFT/IFT ROUND TRIP", img_back)
    cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
    cv2.imshow("FILTERED2 DFT/IFT ROUND TRIP", img_filtered2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # write result to disk
    cv2.imwrite("dft_numpy_roundtrip.png", img_back)
    cv2.imwrite("dft_numpy_highboost_filtered1.png", img_filtered)
    cv2.imwrite("dft_numpy_highboost_filtered2.png", img_filtered2)


if __name__ == "__main__":
    img = "../data/images/2011.JPG"
    low_pass(img)
    high_pass_edge(img)
    high_boost(img)

