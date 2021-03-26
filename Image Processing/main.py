from PIL import Image
import numpy as np
import cmath
import math
import cv2

def transpose(im):
	N = len(im)
	res = []

	for i in range(N):
		res.append([0] * N)
		for j in range(N):
			res[i][j] = im[j][i]
	return res

def dct1d(im):
	N = len(im)
	res = [0] * N
	for i in range(N):
		for j in range(N):
			res[i] += im[j] * math.cos(math.pi * i * (j + 1 / 2) / N)
		if (i != 0):
			res[i] = math.sqrt(2/N) * res[i]
		else:
			res[i] = math.sqrt(1/N) * res[i]
	return res

def dct2d(im):
	N = len(im)

	res= []
	for i in range(N):
		res.append(dct1d(im[i]))
	trans = transpose(res)
	res= []
	for i in range(N):
		res.append(dct1d(trans[i]))
	final = transpose(res)
	return np.array(final)

def idct1d(coef):
	N = len(coef)
	res = [0] * N
	for i in range(N):
		for j in range(N):
			if (j == 0):
				c = math.sqrt(1 / N)
			else:
				c = math.sqrt(2 / N)
			res[i] += c * coef[j] * math.cos(math.pi * j * (i + 1 / 2) / N)
	return res

def idct2d(coef):
	N = len(coef)

	res= []
	for i in range(N):
		res.append(idct1d(coef[i]))
	trans = transpose(res)
	res= []
	for i in range(N):
		res.append(idct1d(trans[i]))
	final = transpose(res)
	return np.array(final)

def fft1d(im):
	N = len(im)
	res = [0] * N
	for i in range(N):
		for k in range(N):
			res[i] += im[k] * cmath.exp(-(2j * cmath.pi * i * k / N))
		res[i] = res[i]
	return res

def fft2d(im):
	N = len(im)

	res= []
	for i in range(N):
		res.append(fft1d(im[i]))
	trans = transpose(res)
	res= []
	for i in range(N):
		res.append(fft1d(trans[i]))
	final = transpose(res)

	return np.array(final)

def ifft1d(coef):
	N = len(coef)
	res = [0] * N
	for i in range(N):
		for k in range(N):
			res[i] += coef[k] * cmath.exp((2j * cmath.pi * i * k / N))
		res[i] = res[i] / N
	return res

def ifft2d(coef): 
	N = len(coef)

	res= []
	for i in range(N):
		res.append(ifft1d(coef[i]))
	trans = transpose(res)
	res= []
	for i in range(N):
		res.append(ifft1d(trans[i]))
	final = transpose(res)
	return np.array(final)



def cv2_dct(fileName):
	print("Starting cv2 DCT.")
	I = cv2.imread(fileName, 0)
	I_array = I.astype(np.float32)

	dct_array = cv2.dct(I_array)

	im = Image.fromarray(dct_array)
	im = im.convert("L")
	im.save("correct_dct_" + fileName + ".png")

	idct_array = cv2.idct(dct_array).astype(np.uint8)

	im = Image.fromarray(idct_array)
	im = im.convert("L")
	im.save("correct_idct_" + fileName + ".png")
	print("Succeed.")

def numpy_fft(fileName):
	print("Starting numpy FFT.")
	I = cv2.imread(fileName, 0)
	I_array = I.astype(np.float32)

	fft_array = np.fft.fft2(I_array)

	im = Image.fromarray(fft_array.real)
	im = im.convert("L")
	im.save("correct_fft_" + fileName + ".png")

	ifft_array = np.fft.ifft2(fft_array)

	im = Image.fromarray(ifft_array.real)
	im = im.convert("L")
	im.save("correct_ifft_" + fileName + ".png")
	print("Succeed.")



def customized_dct(fileName):
	I = cv2.imread(fileName, 0)
	I_array = I.astype(np.float32)

	print("Starting DCT.")
	dct_array = dct2d(I_array).astype(np.float32)

	print("DCT completed.")
	print("Starting creating DCT image.")
	im = Image.fromarray(dct_array)
	im = im.convert("L")
	im.save("customized_dct_" + fileName + ".png")
	
	print("Starting IDCT.")
	idct_array = idct2d(dct_array)

	print("IDCT completed.")
	print("Starting creating IDCT image.")
	im = Image.fromarray(idct_array)
	im = im.convert("L")
	im.save("customized_idct_" + fileName + ".png")

	print("Succeed.\n")

def customized_fft(fileName):
	I = cv2.imread(fileName, 0)
	I_array = I.astype(np.float32)

	print("Starting FFT.")
	fft_array = fft2d(I_array)

	print("FFT completed.")
	print("Starting creating FFT image.")
	im = Image.fromarray(fft_array.real)
	im = im.convert("L")
	im.save("customized_fft_" + fileName + ".png")

	print("Starting IFFT.")
	ifft_array = ifft2d(fft_array)

	print("IFFT completed.")
	print("Starting creating IFFT image.")
	im = Image.fromarray(ifft_array.real)
	im = im.convert("L")
	im.save("customized_ifft_" + fileName + ".png")

	print("Succeed.\n")

def testDCT():
	print("Test 1 of DCT:")
	array = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]).astype(np.float32)
	print("The original array:")
	print(array)
	print("\nThe result of cv2 DCT:")
	print(cv2.dct(array))
	print("\nThe result of customized DCT:")
	print(dct2d(array))
	print("\n-------------------------\n")

def testIDCT():
	print("Test 2 of IDCT:")
	array = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]).astype(np.float32)
	print("\nThe expected output:")
	print(array)
	print("\nThe result of cv2 DCT:")
	print(cv2.idct(cv2.dct(array)))
	print("\nThe result of customized DCT:")
	print(idct2d(dct2d(array)))
	print("\n-------------------------\n")


def testFFT():
	print("Test 3 of FFT:")
	array = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]).astype(np.float32)
	print("\nThe original array:")
	print(array)
	print("\nThe result of numpy FFT:")
	print(np.fft.fft2(array))
	print("\nThe result of customized DCT:")
	print(fft2d(array))
	print("\n-------------------------\n")

def testIFFT():
	print("Test 4 of IFFT:")
	array = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]).astype(np.float32)
	print("\nThe expected output:")
	print(array)
	print("\nThe result of numpy IFFT:")
	print(np.fft.ifft2(np.fft.fft2(array)))
	print("\nThe result of customized IFFT:")
	print(ifft2d(fft2d(array)))
	print("\n-------------------------\n")

def testDCTOnImage(fileName):
	print("Test 5 of DCT on image:")
	cv2_dct(fileName)
	customized_dct(fileName)
	print("\n-------------------------\n")

def testFFTOnImage(fileName):
	print("Test 6 of FFT on image:")
	numpy_fft(fileName)
	customized_fft(fileName)
	print("\n-------------------------\n")