#%%
import tifffile as tiff
import numpy as np
from tkinter import filedialog
from numba import jit, prange

from timeit import default_timer as timer  
from time import time 

def timer_func(func): 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print((t2-t1)) #f'Function {func.__name__!r} executed in {(t2-t1):.4f} s'
        return result 
    return wrap_func 

#playing with parallelized sort networks
@jit(nopython=True)
def compare_and_swap(i, j):
	if i > j:
		return j,i
	else:
		return i,j
	

# @jit(nopython=True)
# def sort3(a):
# 	a[0], a[2] = compare_and_swap(a[0], a[2])
# 	a[0], a[1] = compare_and_swap(a[0], a[1])
# 	a[1], a[2] = compare_and_swap(a[1], a[2])
# 	return a

@jit(nopython=True)
def sort5(a):
# Sort network for 5 element
	for j in prange(2):
		if j==0:
			a[0], a[3] = compare_and_swap(a[0], a[3])
		else:
			a[1], a[4] = compare_and_swap(a[1], a[4])
	for j in prange(2):
		if j==0:
			a[0], a[2] = compare_and_swap(a[0], a[2])
		else:
			a[1], a[3] = compare_and_swap(a[1], a[3])
	for j in prange(2):
		if j==0:
			a[0], a[1] = compare_and_swap(a[0], a[1])
		else:
			a[2], a[4] = compare_and_swap(a[2], a[4])
	for j in prange(2):
		if j==0:
			a[1], a[2] = compare_and_swap(a[1], a[2])
		else:
			a[3], a[4] = compare_and_swap(a[3], a[4])
		
	a[2], a[3] = compare_and_swap(a[2], a[3])
	return a

@jit(nopython=True)
def sort7(a):
	# Sort network for 7 elements
	for j in prange(3):
		if j==0:
			a[0], a[6] = compare_and_swap(a[0], a[6])
		elif j==1:
			a[2], a[3] = compare_and_swap(a[2], a[3])
		else:
			a[4], a[5] = compare_and_swap(a[4], a[5])
	for j in prange(3):
		if j==0:
			a[0], a[2] = compare_and_swap(a[0], a[2])
		elif j==1:
			a[1], a[4] = compare_and_swap(a[1], a[4])
		else:
			a[3], a[6] = compare_and_swap(a[3], a[6])
	for j in prange(3):
		if j==0:
			a[0], a[1] = compare_and_swap(a[0], a[1])
		elif j==1:
			a[2], a[5] = compare_and_swap(a[2], a[5])
		else:
			a[3], a[4] = compare_and_swap(a[3], a[4])
	for i in prange(3):
		if i==0:
			a[1], a[2] = compare_and_swap(a[1], a[2])
		else:
			a[4], a[6] = compare_and_swap(a[4], a[6])
	for j in prange(3):
		if j==0:
			a[2], a[3] = compare_and_swap(a[2], a[3])
		else:
			a[4], a[5] = compare_and_swap(a[4], a[5])
	for i in prange(3):
		if i==0:
			a[1], a[2] = compare_and_swap(a[1], a[2])
		elif i==1:
			a[3], a[4] = compare_and_swap(a[3], a[4])
		else:
			a[5], a[6] = compare_and_swap(a[5], a[6])
	return a

@jit(nopython=True)
def sort8(a):
	for j in prange(4):
		if j==0:
			a[0], a[2] = compare_and_swap(a[0], a[2])
		elif j==1:
			a[1], a[3] = compare_and_swap(a[1], a[3])
		elif j==2:
			a[4], a[6] = compare_and_swap(a[4], a[6])
		else:
			a[5], a[7] = compare_and_swap(a[5], a[7])
	for j in prange(4):
		if j==0:
			a[0], a[4] = compare_and_swap(a[0], a[4])
		elif j==1:
			a[1], a[5] = compare_and_swap(a[1], a[5])
		elif j==2:
			a[2], a[6] = compare_and_swap(a[2], a[6])
		else:
			a[3], a[7] = compare_and_swap(a[3], a[7])
	for j in prange(4):
		if j==0:
			a[0], a[1] = compare_and_swap(a[0], a[1])
		elif j==1:
			a[2], a[3] = compare_and_swap(a[2], a[3])
		elif j==2:
			a[4], a[5] = compare_and_swap(a[4], a[5])
		else:
			a[6], a[7] = compare_and_swap(a[6], a[7])
	for i in prange(2):
		if i==0:
			a[2], a[4] = compare_and_swap(a[2], a[4])
		else:
			a[3], a[5] = compare_and_swap(a[3], a[5])
	for j in prange(2):
		if j==0:
			a[1], a[4] = compare_and_swap(a[1], a[4])
		else:
			a[3], a[6] = compare_and_swap(a[3], a[6])
	for i in prange(3):
		if i==0:
			a[1], a[2] = compare_and_swap(a[1], a[2])
		elif i==1:
			a[3], a[4] = compare_and_swap(a[3], a[4])
		else:
			a[5], a[6] = compare_and_swap(a[5], a[6])
	return a

@jit(nopython=True)
def sort9(a):
	for j in prange(4):
		if j==0:
			a[0], a[3] = compare_and_swap(a[0], a[3])
		elif j==1:
			a[1], a[7] = compare_and_swap(a[1], a[7])
		elif j==2:
			a[2], a[5] = compare_and_swap(a[2], a[5])
		else:
			a[4], a[8] = compare_and_swap(a[4], a[8])
	for j in prange(4):
		if j==0:
			a[0], a[7] = compare_and_swap(a[0], a[7])
		elif j==1:
			a[2], a[4] = compare_and_swap(a[2], a[4])
		elif j==2:
			a[3], a[8] = compare_and_swap(a[3], a[8])
		else:
			a[5], a[6] = compare_and_swap(a[5], a[6])
	for j in prange(4):
		if j==0:
			a[0], a[2] = compare_and_swap(a[0], a[2])
		elif j==1:
			a[1], a[3] = compare_and_swap(a[1], a[3])
		elif j==2:
			a[4], a[5] = compare_and_swap(a[4], a[5])
		else:
			a[7], a[8] = compare_and_swap(a[7], a[8])
	for i in prange(3):
		if i==0:
			a[1], a[4] = compare_and_swap(a[1], a[4])
		elif i==1:
			a[3], a[6] = compare_and_swap(a[3], a[6])
		else:
			a[5], a[7] = compare_and_swap(a[5], a[7])
	for j in prange(4):
		if j==0:
			a[0], a[1] = compare_and_swap(a[0], a[1])
		elif j==1:
			a[2], a[4] = compare_and_swap(a[2], a[4])
		elif j==2:
			a[3], a[5] = compare_and_swap(a[3], a[5])
		else:
			a[6], a[8] = compare_and_swap(a[6], a[8])
	for i in prange(3):
		if i==0:
			a[2], a[3] = compare_and_swap(a[2], a[3])
		elif i==1:
			a[4], a[5] = compare_and_swap(a[4], a[5])
		else:
			a[6], a[7] = compare_and_swap(a[6], a[7])
	for j in prange(3):
		if j==0:
			a[1], a[2] = compare_and_swap(a[1], a[2])
		elif j==1:
			a[3], a[4] = compare_and_swap(a[3], a[4])
		else:
			a[5], a[6] = compare_and_swap(a[5], a[6])
	return a


@jit(nopython=True)
def median(arr):
	length = len(arr)
	if length == 5:
		arr_sorted = sort5(arr)
	elif length == 7:
		arr_sorted = sort7(arr)
	elif length == 8:
		arr_sorted = sort8(arr)
	elif length == 20:
		pass



	# match length:
	# 	case 5:
	# 		arr_sorted = sort5(arr)
	# 	case 7:
	# 		arr_sorted = sort7(arr)
	# 	case 8:
	# 		arr_sorted = sort8(arr)
			
		
	if length % 2 == 0:
		median_value = int(arr_sorted[int(length/2-1)]/2+arr_sorted[int(length/2)]/2)
	else:
		median_value = int(arr_sorted[int((length - 1) / 2)])

	return median_value

# Helpers to get pixel with edge handling
@jit(nopython=True)
def get_pixel_2D(stack, y_, x_):

	if y_ < 0:
		y_ = 0
	elif y_ > stack.shape[0] - 1:
		y_ = stack.shape[0] - 1

	if x_ < 0:
		x_ = 0
	elif x_ > stack.shape[1] - 1:
		x_ = stack.shape[1] - 1

	yi = int(y_)
	xi = int(x_)	

	return stack[yi, xi]

@jit(nopython=True)
def get_pixel_3D(stack,z_, y_, x_):
	# ensure indices are integers and clamp to valid ranges

	if z_ < 0:
		z_ = 0
	elif z_ > stack.shape[0] - 1:
		z_ = stack.shape[0] - 1

	zi = int(z_)

	return get_pixel_2D(stack[zi], y_, x_)

@timer_func
@jit(nopython=True, parallel=True)
def hybrid_2d_median_filter(stack, include_center_pixel=True, filtersize=3):
	stack = stack.astype(np.uint16)  # ensure input is uint16
	height, width = stack.shape
	filtered_stack = np.zeros_like(stack).astype(np.uint16)

	for y in prange(height):
		for x in prange(width):
			# 2D PLUS kernel (center row/col) -> use fixed-size numpy arrays for Numba
			if filtersize == 3:
				marraythisP = np.zeros(5, dtype=np.uint16)
			if filtersize == 5:
				marraythisP = np.zeros(9, dtype=np.uint16)
			marraythisP[0] = get_pixel_2D(stack, y - 1, x)
			marraythisP[1] = get_pixel_2D(stack, y, x - 1)
			marraythisP[2] = get_pixel_2D(stack, y, x)
			marraythisP[3] = get_pixel_2D(stack, y, x + 1)
			marraythisP[4] = get_pixel_2D(stack, y + 1, x)
			if filtersize == 5:
				marraythisP[5] = get_pixel_2D(stack, y - 2, x)
				marraythisP[6] = get_pixel_2D(stack, y + 2, x)
				marraythisP[7] = get_pixel_2D(stack, y, x + 2)
				marraythisP[8] = get_pixel_2D(stack, y, x - 2)

			# 2D X kernel
			if filtersize == 3:
				marraythisX = np.zeros(5, dtype=np.uint16)
			if filtersize == 5:
				marraythisX = np.zeros(9, dtype=np.uint16)
			marraythisX[0] = get_pixel_2D(stack, y - 1, x - 1)
			marraythisX[1] = get_pixel_2D(stack, y - 1, x + 1)
			marraythisX[2] = get_pixel_2D(stack, y, x)
			marraythisX[3] = get_pixel_2D(stack, y + 1, x - 1)
			marraythisX[4] = get_pixel_2D(stack, y + 1, x + 1)
			if filtersize == 5:
				marraythisX[5] = get_pixel_2D(stack, y - 2, x - 2)
				marraythisX[6] = get_pixel_2D(stack, y - 2, x + 2)
				marraythisX[7] = get_pixel_2D(stack, y + 2, x - 2)
				marraythisX[8] = get_pixel_2D(stack, y + 2, x + 2)
			# prepare medianarray (2 entries, optionally 3)
			if include_center_pixel:
				medianarray = np.zeros(3, dtype=np.uint16)
				medianarray[2] = get_pixel_2D(stack, y, x)
			else:
				medianarray = np.zeros(2, dtype=np.uint16)
			medianarray[0] = median(marraythisX)
			medianarray[1] = median(marraythisP)
			filtered_stack[y, x] = median(medianarray)
	return filtered_stack
		

# @timer_func
@jit(nopython=True, parallel=True)
def hybrid_3d_median_filter(stack, include_center_pixel=False):
	stack = stack.astype(np.uint16) 
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
				marraythisP[0] = get_pixel_3D(stack, z, y - 1, x)
				marraythisP[1] = get_pixel_3D(stack, z, y, x - 1)
				marraythisP[2] = get_pixel_3D(stack, z, y, x)
				marraythisP[3] = get_pixel_3D(stack, z, y, x + 1)
				marraythisP[4] = get_pixel_3D(stack, z, y + 1, x)
				# 2D X kernel
				marraythisX = np.zeros(5, dtype=np.uint16)
				marraythisX[0] = get_pixel_3D(stack, z, y - 1, x - 1)
				marraythisX[1] = get_pixel_3D(stack, z, y - 1, x + 1)
				marraythisX[2] = get_pixel_3D(stack, z, y, x)
				marraythisX[3] = get_pixel_3D(stack, z, y + 1, x - 1)
				marraythisX[4] = get_pixel_3D(stack, z, y + 1, x + 1)
				# 3D PLUS kernel
				marray3P = np.zeros(7, dtype=np.uint16)
				marray3P[0] = get_pixel_3D(stack, before_z, y, x)
				marray3P[1] = get_pixel_3D(stack, z, y - 1, x)
				marray3P[2] = get_pixel_3D(stack, z, y, x - 1)
				marray3P[3] = get_pixel_3D(stack, z, y, x)
				marray3P[4] = get_pixel_3D(stack, z, y, x + 1)
				marray3P[5] = get_pixel_3D(stack, z, y + 1, x)
				marray3P[6] = get_pixel_3D(stack, after_z, y, x)
				# 3D X kernels
				marray3Xa = np.zeros(5, dtype=np.uint16)
				marray3Xa[0] = get_pixel_3D(stack, before_z, y - 1, x - 1)
				marray3Xa[1] = get_pixel_3D(stack, after_z, y + 1, x + 1)
				marray3Xa[2] = get_pixel_3D(stack, z, y, x)
				marray3Xa[3] = get_pixel_3D(stack, before_z, y + 1, x - 1)
				marray3Xa[4] = get_pixel_3D(stack, after_z, y - 1, x + 1)
				marray3Xb = np.zeros(5, dtype=np.uint16)
				marray3Xb[0] = get_pixel_3D(stack, before_z, y - 1, x)
				marray3Xb[1] = get_pixel_3D(stack, after_z, y + 1, x)
				marray3Xb[2] = get_pixel_3D(stack, z, y, x)
				marray3Xb[3] = get_pixel_3D(stack, before_z, y + 1, x)
				marray3Xb[4] = get_pixel_3D(stack, after_z, y - 1, x)
				marray3Xc = np.zeros(5, dtype=np.uint16)
				marray3Xc[0] = get_pixel_3D(stack, before_z, y - 1, x + 1)
				marray3Xc[1] = get_pixel_3D(stack, after_z, y + 1, x - 1)
				marray3Xc[2] = get_pixel_3D(stack, z, y, x)
				marray3Xc[3] = get_pixel_3D(stack, before_z, y + 1, x + 1)
				marray3Xc[4] = get_pixel_3D(stack, after_z, y - 1, x - 1)
				marray3Xd = np.zeros(5, dtype=np.uint16)
				marray3Xd[0] = get_pixel_3D(stack, before_z, y, x - 1)
				marray3Xd[1] = get_pixel_3D(stack, after_z, y, x + 1)
				marray3Xd[2] = get_pixel_3D(stack, z, y, x)
				marray3Xd[3] = get_pixel_3D(stack, before_z, y, x + 1)
				marray3Xd[4] = get_pixel_3D(stack, after_z, y, x - 1)

				# prepare medianarray (7 entries, optionally 8)
				if include_center_pixel:
					medianarray = np.zeros(8, dtype=np.uint16)
					medianarray[7] = get_pixel_3D(stack, z, y, x)
				else:
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

if __name__ == "__main__":

	# filepath = filedialog.askopenfilename()
	# with tiff.TiffFile(filepath) as tif:
	# 	print('Reading image...')
	# 	image = tif.asarray()
	# 	print('Image shape:', image.shape)

	if len(image.shape) ==2:
		filtered_image = hybrid_2d_median_filter(image, include_center_pixel=False, filtersize=5)
		tiff.imwrite('denoised_image.tiff', filtered_image, 
				compression='zlib',
				compressionargs={'level': 6},
				photometric='minisblack',
				metadata={'axes': 'YX'})

	if len(image.shape) == 3:
		filtered_stack = hybrid_3d_median_filter(image)
		# tiff.imwrite('denoised_image.tiff', filtered_stack, 
		# 		compression='zlib',
		# 		compressionargs={'level': 6},
		# 		photometric='minisblack',
		# 		metadata={'axes': 'ZYX'})
	elif len(image.shape) == 4:
		for volume in range(image.shape[0]):
			print(f'Processing volume {volume+1}/{image.shape[0]}')
			image[volume] = hybrid_3d_median_filter(image[volume])
		# tiff.imwrite('denoised_image.tiff', image,
		# 		compression='zlib',
		# 		compressionargs={'level': 6},
		# 		photometric='minisblack',
		# 		metadata={'axes': 'TZYX'})

# %%
