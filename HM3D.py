#%%
import tifffile as tiff
import numpy as np
from tkinter import filedialog
from numba import jit, prange

#%%
@jit(nopython=True)
def median(arr):
	arr_sorted = np.sort(arr).astype(np.uint16)
	length = len(arr_sorted)
	if length % 2 == 0:
		median_value = int(arr_sorted[int(length/2-1)]/2+arr_sorted[int(length/2)]/2)
	else:
		median_value = int(arr_sorted[int((length - 1) / 2)])

	# if length % 2 == 0:
	# 	median_value = (arr_sorted[int(length/2-1)]+arr_sorted[int(length/2)])/2
	# else:
	# 	median_value = arr_sorted[int((length-1)/2)]
	
	return median_value

# Helper to get pixel with edge handling
@jit(nopython=True)
def get_pixel(stack,z_, y_, x_):
	# ensure indices are integers and clamp to valid ranges

	if z_ < 0:
		z_ = 0
	elif z_ > stack.shape[0] - 1:
		z_ = stack.shape[0] - 1

	if y_ < 0:
		y_ = 0
	elif y_ > stack.shape[1] - 1:
		y_ = stack.shape[1] - 1

	if x_ < 0:
		x_ = 0
	elif x_ > stack.shape[2] - 1:
		x_ = stack.shape[2] - 1

	zi = int(z_)
	yi = int(y_)
	xi = int(x_)	

	return stack[zi, yi, xi]
	
@jit(nopython=True, parallel=True)
def hybrid_3d_median_filter(stack, include_center_pixel=False):
	stack = stack.astype(np.uint16)  # ensure input is uint16
	# stack: 3D numpy array (depth, height, width)
	depth, height, width = stack.shape
	filtered_stack = np.zeros_like(stack).astype(np.uint16)
	
	for z in prange(depth):
		# compute neighbor z indices with integer arithmetic
		before_z = z - 1
		if before_z < 0:
			before_z = 0
		after_z = z + 1
		if after_z > depth - 1:
			after_z = depth - 1
		for y in prange(height):
			for x in prange(width):
				# 2D PLUS kernel (center row/col) -> use fixed-size numpy arrays for Numba
				marraythisP = np.zeros(5, dtype=np.uint16)
				marraythisP[0] = get_pixel(stack, z, y - 1, x)
				marraythisP[1] = get_pixel(stack, z, y, x - 1)
				marraythisP[2] = get_pixel(stack, z, y, x)
				marraythisP[3] = get_pixel(stack, z, y, x + 1)
				marraythisP[4] = get_pixel(stack, z, y + 1, x)
				# 2D X kernel
				marraythisX = np.zeros(5, dtype=np.uint16)
				marraythisX[0] = get_pixel(stack, z, y - 1, x - 1)
				marraythisX[1] = get_pixel(stack, z, y - 1, x + 1)
				marraythisX[2] = get_pixel(stack, z, y, x)
				marraythisX[3] = get_pixel(stack, z, y + 1, x - 1)
				marraythisX[4] = get_pixel(stack, z, y + 1, x + 1)
				# 3D PLUS kernel
				marray3P = np.zeros(7, dtype=np.uint16)
				marray3P[0] = get_pixel(stack, before_z, y, x)
				marray3P[1] = get_pixel(stack, z, y - 1, x)
				marray3P[2] = get_pixel(stack, z, y, x - 1)
				marray3P[3] = get_pixel(stack, z, y, x)
				marray3P[4] = get_pixel(stack, z, y, x + 1)
				marray3P[5] = get_pixel(stack, z, y + 1, x)
				marray3P[6] = get_pixel(stack, after_z, y, x)
				# 3D X kernels
				marray3Xa = np.zeros(5, dtype=np.uint16)
				marray3Xa[0] = get_pixel(stack, before_z, y - 1, x - 1)
				marray3Xa[1] = get_pixel(stack, after_z, y + 1, x + 1)
				marray3Xa[2] = get_pixel(stack, z, y, x)
				marray3Xa[3] = get_pixel(stack, before_z, y + 1, x - 1)
				marray3Xa[4] = get_pixel(stack, after_z, y - 1, x + 1)
				marray3Xb = np.zeros(5, dtype=np.uint16)
				marray3Xb[0] = get_pixel(stack, before_z, y - 1, x)
				marray3Xb[1] = get_pixel(stack, after_z, y + 1, x)
				marray3Xb[2] = get_pixel(stack, z, y, x)
				marray3Xb[3] = get_pixel(stack, before_z, y + 1, x)
				marray3Xb[4] = get_pixel(stack, after_z, y - 1, x)
				marray3Xc = np.zeros(5, dtype=np.uint16)
				marray3Xc[0] = get_pixel(stack, before_z, y - 1, x + 1)
				marray3Xc[1] = get_pixel(stack, after_z, y + 1, x - 1)
				marray3Xc[2] = get_pixel(stack, z, y, x)
				marray3Xc[3] = get_pixel(stack, before_z, y + 1, x + 1)
				marray3Xc[4] = get_pixel(stack, after_z, y - 1, x - 1)
				marray3Xd = np.zeros(5, dtype=np.uint16)
				marray3Xd[0] = get_pixel(stack, before_z, y, x - 1)
				marray3Xd[1] = get_pixel(stack, after_z, y, x + 1)
				marray3Xd[2] = get_pixel(stack, z, y, x)
				marray3Xd[3] = get_pixel(stack, before_z, y, x + 1)
				marray3Xd[4] = get_pixel(stack, after_z, y, x - 1)

				# prepare medianarray (7 entries, optionally 8)
				# if include_center_pixel:
				# 	medianarray = np.zeros(8, dtype=np.uint16)
				# 	medianarray[7] = get_pixel(stack, z, y, x)
				# else:
				medianarray = np.zeros(7, dtype=np.uint16)
				medianarray[0] = median(marraythisX)
				medianarray[1] = median(marraythisP)
				medianarray[2] = median(marray3P)
				medianarray[3] = median(marray3Xa)
				medianarray[4] = median(marray3Xb)
				medianarray[5] = median(marray3Xc)
				medianarray[6] = median(marray3Xd)

				filtered_stack[z, y, x] = median(medianarray)
	return filtered_stack

#%%
# filepath = filedialog.askopenfilename()
# with tiff.TiffFile(filepath) as tif:
# 	print('Reading image...')
# 	image = tif.asarray()
# 	print('Image shape:', image.shape)
#%%
if len(image.shape) == 3:
	filtered_stack = hybrid_3d_median_filter(image)
	tiff.imwrite('denoised_image.tiff', filtered_stack, 
			  compression='zlib',
			  compressionargs={'level': 6},
			  photometric='minisblack',
			  metadata={'axes': 'ZYX'})
elif len(image.shape) == 4:
	for volume in range(image.shape[0]):
		print(f'Processing volume {volume+1}/{image.shape[0]}')
		image[volume] = hybrid_3d_median_filter(image[volume])
	tiff.imwrite('denoised_image.tiff', image,
			  compression='zlib',
			  compressionargs={'level': 6},
			  photometric='minisblack',
			  metadata={'axes': 'TZYX'})

# tiff.imwrite('denoised_image.tiff', filtered_stack) # save the denoising image as tiff

# %%
