from numba import jit, prange
import numpy as np

### Functions for paralell processing ###
#####CPU-processing#####
@jit(parallel=True)
def flatten_4D(data):
    # create new array for the flattened image
    flattened_image = np.zeros((data.shape[0], data.shape[2], data.shape[3]), dtype='uint16')
    
    # parallelize the loop over planes and rows
    for time in prange(data.shape[0]):
        for plane in prange(data.shape[1]):
            flattened_image[time, :, :] += data[time, plane, :, :]            
    return flattened_image

@jit(parallel=True)  
def remapping3D(data,shape_array,factor,FDML=False): # factor must be in (2,4,8,16,32,...)
    # calculate new row count
    new_row_count = data.shape[1] * factor
    
    # create new array for the zoomed image
    zoomed_image = np.zeros((data.shape[0], new_row_count, data.shape[2]), dtype='uint16')
    
    # parallelize the loop over planes and rows
    for plane in prange(data.shape[0]):
        for row in prange(data.shape[1]):
            # calculate the start index for the interpolated rows
            start = row * factor
            
            # fill the start row with the original data
            zoomed_image[plane, start, :] = data[plane, row, :]
            
            # interpolate between the original rows
            if factor > 1 and row < data.shape[1] - 1:
                next_row = data[plane, row + 1, :]
                for i in range(1, factor):
                    alpha = i / factor
                    zoomed_image[plane, start + i, :] = (
                        (1 - alpha) * data[plane, row, :] + alpha * next_row
                    ).astype('uint16')
            
            # fill the end row with the original data
            elif factor > 1 and row == data.shape[1] - 1:
                for i in range(1, factor):
                    zoomed_image[plane, start + i, :] = data[plane, row, :]
    data=zoomed_image

    dim=shape_array.shape[1]
    dim_original = data.shape[1]
    remapped_image = np.zeros((data.shape[0],dim,data.shape[2]),dtype='uint16')
    for plane in prange(data.shape[0]):
        sum_correction_factor = 0
        for row in range(dim):
            if FDML:
                correction_factor = (1/(np.pi*np.sqrt(1.5*dim-row)*np.sqrt(row+1/2*dim)))/(1/3)
            else:
                correction_factor = 1/(np.pi*np.sqrt(-1*(row+1/2)*(row+1/2-dim)))

            sum_correction_factor += correction_factor
            upsampled_row = int(np.round(dim_original*sum_correction_factor))
            bins= int(np.round(dim_original*correction_factor))
            for pixel in prange(data.shape[2]):
                remapped_image[plane,row,pixel] = np.mean(data[plane,upsampled_row:upsampled_row+bins,pixel])
    data = remapped_image
    return data


@jit(parallel=True)
def remapping2D(data,shape_array,factor,FDML=False): 
    new_row_count = data.shape[0] * factor
    
    # create new array for the zoomed image
    zoomed_image = np.zeros((new_row_count, data.shape[1]), dtype='uint16')
    
    # parallelize the loop over rows
    for row in prange(data.shape[0]):
        # calculate the start index for the interpolated rows
        start = row * factor
        
        # fill the start row with the original data
        zoomed_image[start, :] = data[row, :]
        
        # interpolate between the original rows alpha is the weight depending on proximity to the original row
        if factor > 1 and row < data.shape[0] - 1:
            next_row = data[row + 1, :]
            for i in range(1, factor):
                alpha = i / factor
                zoomed_image[start + i, :] = (
                    (1 - alpha) * data[row, :] + alpha * next_row
                ).astype('uint16')
        
        # fill the end row with the original data
        elif factor > 1 and row == data.shape[0] - 1:
            for i in range(1, factor):
                zoomed_image[start + i, :] = data[row, :]
    
    data = zoomed_image
    dim=shape_array.shape[0]
    dim_original = data.shape[0]
    remapped_image = np.zeros((dim,data.shape[1]),dtype='uint16')
    sum_correction_factor = 0
    for row in range(dim):
        if FDML:
            correction_factor = (1/(np.pi*np.sqrt(1.5*dim-row)*np.sqrt(row+1/2*dim)))/(1/3)
        else:
            correction_factor = 1/(np.pi*np.sqrt(-1*(row+1/2)*(row+1/2-dim)))
        sum_correction_factor += correction_factor
        upsampled_row = int(np.round(dim_original*sum_correction_factor))
        bins= int(np.round(dim_original*correction_factor))
        for pixel in prange(data.shape[1]):      
            remapped_image[row,pixel] = np.mean(data[upsampled_row:upsampled_row+bins,pixel])
    return remapped_image