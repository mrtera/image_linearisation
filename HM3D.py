#%%
import tifffile as tiff
import numpy as np
from tkinter import filedialog
from numba import jit, prange
#%%

def median(arr):
	arr_sorted = np.sort(arr)
	length = len(arr_sorted)
	if length % 2 == 0:
		return (arr_sorted[length // 2 - 1] + arr_sorted[length // 2]) / 2.0
	else:
		return arr_sorted[(length - 1) // 2]
	
# @jit(parallel=True)
def hybrid_3d_median_filter(stack, include_center_pixel=False):
	# stack: 3D numpy array (depth, height, width)
	depth, height, width = stack.shape
	filtered_stack = np.zeros_like(stack)
	
	for z in range(depth):
		before_z = max(z - 1, 0)
		after_z = min(z + 1, depth - 1)
		for y in range(height):
			for x in range(width):
				# Helper to get pixel with edge handling
				def get_pixel(z_, y_, x_):
					z_ = min(max(z_, 0), depth - 1)
					y_ = min(max(y_, 0), height - 1)
					x_ = min(max(x_, 0), width - 1)
					return stack[z_, y_, x_]
				
				# 2D PLUS kernel (center row/col)
				marraythisP = [
					get_pixel(z, y - 1, x),
					get_pixel(z, y, x - 1),
					get_pixel(z, y, x),
					get_pixel(z, y, x + 1),
					get_pixel(z, y + 1, x)
				]
				# 2D X kernel
				marraythisX = [
					get_pixel(z, y - 1, x - 1),
					get_pixel(z, y - 1, x + 1),
					get_pixel(z, y, x),
					get_pixel(z, y + 1, x - 1),
					get_pixel(z, y + 1, x + 1)
				]
				# 3D PLUS kernel
				marray3P = [
					get_pixel(before_z, y, x),
					get_pixel(z, y - 1, x),
					get_pixel(z, y, x - 1),
					get_pixel(z, y, x),
					get_pixel(z, y, x + 1),
					get_pixel(z, y + 1, x),
					get_pixel(after_z, y, x)
				]
				# 3D X kernels
				marray3Xa = [
					get_pixel(before_z, y - 1, x - 1),
					get_pixel(after_z, y + 1, x + 1),
					get_pixel(z, y, x),
					get_pixel(before_z, y + 1, x - 1),
					get_pixel(after_z, y - 1, x + 1)
				]
				marray3Xb = [
					get_pixel(before_z, y - 1, x),
					get_pixel(after_z, y + 1, x),
					get_pixel(z, y, x),
					get_pixel(before_z, y + 1, x),
					get_pixel(after_z, y - 1, x)
				]
				marray3Xc = [
					get_pixel(before_z, y - 1, x + 1),
					get_pixel(after_z, y + 1, x - 1),
					get_pixel(z, y, x),
					get_pixel(before_z, y + 1, x + 1),
					get_pixel(after_z, y - 1, x - 1)
				]
				marray3Xd = [
					get_pixel(before_z, y, x - 1),
					get_pixel(after_z, y, x + 1),
					get_pixel(z, y, x),
					get_pixel(before_z, y, x + 1),
					get_pixel(after_z, y, x - 1)
				]
				medianarray = [
					median(marraythisX),
					median(marraythisP),
					median(marray3P),
					median(marray3Xa),
					median(marray3Xb),
					median(marray3Xc),
					median(marray3Xd)
				]
				if include_center_pixel:
					medianarray.append(get_pixel(z, y, x))
				filtered_stack[z, y, x] = median(medianarray)
	return filtered_stack

#%%
filepath = filedialog.askopenfilename()
with tiff.TiffFile(filepath) as tif:
	print('Reading image...')
	image = tif.asarray()
	image = image[0]
#%%
filtered_stack = hybrid_3d_median_filter(image)

tiff.imwrite('denoised_image.tiff', filtered_stack) # save the denoising image as tiff

# %%
