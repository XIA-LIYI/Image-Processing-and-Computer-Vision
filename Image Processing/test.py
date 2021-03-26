from main import *
import numpy as np
np.set_printoptions(suppress=True)

## The example array we used to test is [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
### Test 1 DCT ###
testDCT()

### Test 2 IDCT ###
testIDCT()

### Test 3 FFT ###
testFFT()

### Test 4 IFFT ###
testIFFT()

### Test 5 DCT on Image ###
## The test will run the sample picture named as 'sample.png' 
## and produce the result given by cv2 and my customized dct() and idct() function.
## I consider the result given by cv2 as correct output. We can manually check whether
## the output given by customized functon is same.

testDCTOnImage("sample.png")
testDCTOnImage("sample1.png")
testDCTOnImage("sample2.png")

### Test 6 FFT on Image ###
## Same process as Test 5. However, in Test 6, the correct ouput is given by numpy.

testFFTOnImage("sample.png")
testFFTOnImage("sample1.png")
testFFTOnImage("sample2.png")

## The output when producing image may report error due to data type (encoding) problem.
## I make sure that all sample imgaes can be processed properly with python 3.9, latest cv2 and numpy.

### Notes on Algorithms Used ###
## The algorithms used in dct and fft follows the instructions of project file which running 1d version on each row and then on each column.
## It has great improvement compared with direct 2d version.
## It can be improved in efficiency using multi-thread algorithm but I did not implement here. 
