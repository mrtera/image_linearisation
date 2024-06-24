#%%
from PIL import Image, ImageTk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import tifffile as tiff
import numpy as np
import scipy as sp
from numba import jit, prange
from timeit import default_timer as timer   

@jit()    #for GPU acceleration
def remapping1DGPU(remapped_image,zoomed_image,upsampling_factor):
    sum_correction_factor = 0
    dim=remapped_image.shape[0]
    dim_upsampled = zoomed_image.shape[0]
    for row in np.arange(dim):
        correction_factor = 1/(np.pi*np.sqrt(-1*(row+1/2)*(row+1/2-dim)))
        sum_correction_factor += correction_factor
        upsampled_row = int(np.round(dim_upsampled*sum_correction_factor))
        bins= int(np.round(dim*upsampling_factor*correction_factor))

        for pixels in prange(remapped_image.shape[1]):                # GPU computed not a lot of gain 
            remapped_image[row,pixels] = np.mean(zoomed_image[upsampled_row:upsampled_row+bins,pixels])
        
    return remapped_image

def remapping1DCPU(remapped_image,zoomed_image,upsampling_factor):
    sum_correction_factor = 0
    dim=remapped_image.shape[0]
    dim_upsampled = zoomed_image.shape[0]
    for row in np.arange(dim):
        correction_factor = 1/(np.pi*np.sqrt(-1*(row+1/2)*(row+1/2-dim)))
        sum_correction_factor += correction_factor
        upsampled_row = int(np.round(dim_upsampled*sum_correction_factor))
        bins= int(np.round(dim*upsampling_factor*correction_factor))

        remapped_image[row] = np.mean(zoomed_image[upsampled_row:upsampled_row+bins],axis=0) # CPU computed
    return remapped_image
    

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Processing')
        self.root.resizable(True, True)

        self.label = Label(root, text='Upsampleing factor X:')
        self.label.grid(row=0, column=1)
        self.upsampling_factor_X_spinbox = Spinbox(root, from_=1, to=100, width=4)
        self.upsampling_factor_X_spinbox.set(3)
        self.upsampling_factor_X_spinbox.grid(row=0, column=2)

        self.label = Label(root, text='Upsampleing factor Y:')
        self.label.grid(row=1, column=1)
        self.upsampling_factor_Y_spinbox = Spinbox(root, from_=1, to=100, width=4)
        self.upsampling_factor_Y_spinbox.set(3)
        self.upsampling_factor_Y_spinbox.grid(row=1, column=2)
        
        self.label = Label(root, text='Upsampleing factor Z:')
        self.label.grid(row=2, column=1)
        self.upsampling_factor_Z_spinbox = Spinbox(root, from_=1, to=100, width=4)
        self.upsampling_factor_Z_spinbox.set(3)
        self.upsampling_factor_Z_spinbox.grid(row=2, column=2)

        self.snow_threshold_spinbox = Spinbox(root, from_=0, to=0.99, width=4, increment=0.9, format='%.2f')
        self.snow_threshold_spinbox.set(0.8)
        self.snow_threshold_spinbox.grid(row=3, column=2)

        self.remove_snow = BooleanVar(value=True)
        self.remove_snow_checkbox = Checkbutton(root, text='remove snow above x*max', variable=self.remove_snow)
        self.remove_snow_checkbox.grid(row=3, column=1)

        self.is2D_video = BooleanVar(value=False)
        self.is2D_video_checkbox = Checkbutton(root, text='2D Video', variable=self.is2D_video)
        self.is2D_video_checkbox.grid(row=3, column=0,)

        self.do_x_correction = BooleanVar(value=False)
        self.do_x_correction_checkbox = Checkbutton(root, text='X', variable=self.do_x_correction)
        self.do_x_correction_checkbox.grid(row=0, column=0)

        self.do_Y_correction = BooleanVar(value=True)
        self.do_Y_correction_checkbox = Checkbutton(root, text='Y', variable=self.do_Y_correction)
        self.do_Y_correction_checkbox.grid(row=1, column=0)

        self.do_z_correction = BooleanVar(value=True)
        self.do_z_correction_checkbox = Checkbutton(root, text='Z', variable=self.do_z_correction)
        self.do_z_correction_checkbox.grid(row=2, column=0)

        self.try_GPU = BooleanVar(value=False)
        self.try_GPU_checkbox = Checkbutton(root, text='Try GPU', variable=self.try_GPU)
        self.try_GPU_checkbox.grid(row=5, column=1)

        self.open = Button(root, text='Open Image', command=self.open_image)
        self.open.grid(row=5, column=0, columnspan=1)
        self.process = Button(root, text='Process Image', command=self.upsample)
        self.process.grid(row=5, column=2,)

        self.remapped_image = None

    def open_image(self):

        self.filename = filedialog.askopenfilename()
        with tiff.TiffFile(self.filename) as tif:
            self.dim = tif.series[0].ndim
            self.tif_shape = tif.series[0].shape
            self.dtype = tif.pages[0].dtype
            self.axes = tif.series[0].axes
            if self.dim in [2,3,4] and not self.is2D_video.get():
                print('Found Stack dimension: '+str(tif.series[0].shape))
                try:
                    print('t dim = '+str(tif.series[0].shape[-4]))
                    self.t_dim = tif.series[0].shape[-4]
                except IndexError:
                    pass
                try:
                    print('Z dim = '+str(tif.series[0].shape[-3]))
                    self.z_dim = tif.series[0].shape[-3]
                except IndexError:
                    pass
                try:
                    print('Y dim = '+str(tif.series[0].shape[-2]))
                    print('X dim = '+str(tif.series[0].shape[-1]))
                except IndexError:
                    pass
            elif self.dim == 3 and not self.is2D_video.get():
                print('Found Stack dimension: '+str(tif.series[0].shape))
                print('t dim = '+str(tif.series[0].shape[-3]))
                print('Y dim = '+str(tif.series[0].shape[-2]))
                print('X dim = '+str(tif.series[0].shape[-1]))
            else:            
                print('Image dimension not supported') 


    def upsample(self):
        self.upsampling_factor_X = int(self.upsampling_factor_X_spinbox.get())
        self.upsampling_factor_Y = int(self.upsampling_factor_Y_spinbox.get())
        self.upsampling_factor_Z = int(self.upsampling_factor_Z_spinbox.get())
        self.is2D = self.is2D_video.get()
        self.melt = self.remove_snow.get()
        self.snow_threshold = float(self.snow_threshold_spinbox.get())
        if self.dim in [2,3] and not self.is2D_video.get():

            with tiff.TiffFile(self.filename) as tif:
                data = tif.asarray()
                if self.melt:
                    snow_value = np.amax(data)
            if self.dim == 2:
                if self.melt:
                    data = self.melt_snow(data,snow_value,D2=True)
                remapped_image = self.process_2D(data)
                self.save_image(remapped_image)
            elif self.dim == 3:
                if self.melt:
                    data = self.melt_snow(data,snow_value)
                remapped_image = self.process_3D(data)
                self.save_image(remapped_image)

        elif self.dim == 4 and not self.is2D:
            self.process_4D()

        elif self.dim == 3 and self.is2D:
            self.process_2Dt()

        else:
            print('Image dimension not supported!')
        

    def memap(self):
            # create a memmory mapped array to enable processing of larger than RAM files:
            self.memap_filename = self.filename.removesuffix('.tif')+'_processed'+'.tif'
            shape = self.tif_shape
            print('Creating memap file, might take a while, shape: '+str(shape))
            dtype = self.dtype
            # create an empty OME-TIFF file
            tiff.imwrite(self.memap_filename, shape=shape, dtype=dtype, metadata={'axes': self.axes})

            # memory map numpy array to data in OME-TIFF file
            memap_stack = tiff.memmap(self.memap_filename)
            return memap_stack
    
    
    def process_4D(self):
        # Load data either in RAM or as memmap
        memmap = False
        try:
            with tiff.TiffFile(self.filename) as tif:
                stack = tif.asarray()
        except np.core._exceptions._ArrayMemoryError:
            memmap = True
            print('MemoryError: File too large for RAM, processing with memmap')
            stack = self.memap()
            # write data to memory-mapped array    
            print('Writing data to memory-mapped array')
            with tiff.TiffFile(self.filename) as tif:
                for timepoints in range(self.t_dim):
                    for volumes in range(self.z_dim):
                        stack[timepoints,volumes] = tif.pages[timepoints*self.z_dim+volumes].asarray()
                    if timepoints % 50 == 0:
                        print(str(timepoints) + '/' + str(self.t_dim) + ' Volumes written')
            print('Data written to memory-mapped array') 
        
        # melt snow if selected
        if self.melt:
            snow_value = np.amax(stack)
            print('Remvoing snow above '+str(self.snow_threshold*snow_value))
            for timestep in range(self.t_dim):
                stack[timestep] = self.melt_snow(stack[timestep],snow_value)
                if timestep % 50 == 0:
                    print('removed snow in '+str(timestep)+' Volumes')
            print('Snow removed')

        # process data
        print('correcting for sin distorsion')
        for timestep in range(self.t_dim):
            start=timer()
            stack[timestep] = self.process_3D(stack[timestep])
            print('Volume '+str(timestep)+' corrected')
            print('Time elapsed: '+str(timer()-start))
        
        if memmap:
            stack.flush()
            print('Data saved')
        else:
            self.save_image(stack)        
        
        return
    
    
    def process_2Dt(self):
        memmap = False
        try:
            with tiff.TiffFile(self.filename) as tif:
                stack = tif.asarray()
        except np.core._exceptions._ArrayMemoryError:
            stack = self.memap()
            # write data to memory-mapped array
            print('Writing data to memory-mapped array')
            with tiff.TiffFile(self.filename) as tif:
                for timepoints in range(self.z_dim):
                    if timepoints % 100 == 0:
                        print(str(timepoints) + '/' + str(self.z_dim) + ' Frames written')
                    stack[timepoints] = tif.pages[timepoints].asarray()
            print('Data written to memory-mapped array')
        
        # process data in memory-mapped array
        # melt snow 2D if selected
        if self.melt:
            snow_value = np.amax(stack)
            print('Max Snow value: '+str(snow_value) + ' filtering all values above ' + str(self.snow_threshold*snow_value))
            for timestep in np.arange(self.z_dim): 
                stack[timestep] = self.melt_snow(stack[timestep],snow_value,D2=True)
                stack[timestep] = self.process_2D(stack[timestep])
                print('Frame '+str(timestep)+' corrected')
        
        else:
            for timestep in np.arange(self.z_dim): 
                stack[timestep] = self.melt_snow(stack[timestep],snow_value,D2=True)
                stack[timestep] = self.process_2D(stack[timestep])
                print('Frame '+str(timestep)+' corrected')

        if memmap:
            stack.flush() 
            print('Data saved')
        else:
            self.save_image(stack)               
        return
    
    
    def process_3D(self,remapped_image):
        if self.do_z_correction.get():
            for images in range(remapped_image.shape[0]):
                remapped_image[images] = self.remapping2D(remapped_image[images],self.upsampling_factor_Z)

        if self.do_Y_correction.get():
            remapped_image = np.swapaxes(remapped_image,0,1)
            for images in range(remapped_image.shape[0]):
                remapped_image[images] = self.remapping2D(remapped_image[images],self.upsampling_factor_Y)
            remapped_image = np.swapaxes(remapped_image,0,1)
            
        if self.do_x_correction.get():
            remapped_image = np.swapaxes(remapped_image,0,2)
            for images in range(remapped_image.shape[0]):
                remapped_image[images] = self.remapping2D(remapped_image[images],self.upsampling_factor_X)
            remapped_image = np.swapaxes(remapped_image,0,2)
        return remapped_image


    def process_2D(self,data):
        if self.do_Y_correction.get():
            remapped_image = self.remapping2D(data,self.upsampling_factor_Y)
        if self.do_x_correction.get():
            remapped_image = np.swapaxes(data,0,1)
            remapped_image = self.remapping2D(remapped_image,self.upsampling_factor_X)
            remapped_image = np.swapaxes(remapped_image,0,1)
        return remapped_image


### Remapping ###
    def remapping3D(self,remapped_image,upsampling_factor):        
        dim=remapped_image.shape[0]
        sum_correction_factor = 0
        zoomed_image = sp.ndimage.zoom(remapped_image,(upsampling_factor, 1, 1),order=1)
        dim_upsampled = zoomed_image.shape[0]

        for plane in np.arange(dim):
            correction_factor = self.correction_factor(plane,dim)
            sum_correction_factor += correction_factor
            upsampled_plane = np.round(dim_upsampled*sum_correction_factor).astype(int)
            bins= np.round(dim*upsampling_factor*correction_factor).astype(int)
            remapped_image[plane] = np.mean(zoomed_image[upsampled_plane:upsampled_plane+bins],axis=0)
        return remapped_image
    

    def remapping2D(self,remapped_image,upsampling_factor):
        zoomed_image = sp.ndimage.zoom(remapped_image,(upsampling_factor, 1),order=1)
        if self.try_GPU.get():
            remapped_image = remapping1DGPU(remapped_image,zoomed_image,upsampling_factor)
        else:
            remapped_image = remapping1DCPU(remapped_image,zoomed_image,upsampling_factor)
        return remapped_image


    def correction_factor(self,current_index, max_index):
        return 1/(np.pi*np.sqrt(-1*(current_index+1/2)*(current_index+1/2-max_index)))

### Snow removal ###
    def melt_snow(self,data,snow_value,D2=False):
        filtered_data = data
        if D2:
            kernel = np.ones((3,3))/6
            kernel[1,:]=0
            snow_coords = list(zip(*np.where(data > self.snow_threshold*snow_value)))
            for flakes in snow_coords:
                try:
                    filtered_data[flakes] = np.sum(data[flakes[0]-1:flakes[0]+2:2,flakes[1]-1:flakes[1]+2]*kernel).astype('uint16')
                except IndexError:
                    filtered_data[flakes] = 0
                except ValueError:
                    filtered_data[flakes] = 0
                except RuntimeWarning:
                    pass

        else:
            snow_coords = list(zip(*np.where(data > self.snow_threshold*snow_value)))
            kernel = np.ones((3,3,3))/24
            kernel[1,1,:]=0
            for flakes in snow_coords:
                try:
                    filtered_data[flakes] = np.sum(data[flakes[0]-1:flakes[0]+2,flakes[1]-1:flakes[1]+2,flakes[2]-1:flakes[2]+2]*kernel).astype('uint16')
                except IndexError:
                    filtered_data[flakes] = 0
                except ValueError:
                    filtered_data[flakes] = 0
                except RuntimeWarning:
                    pass

        return data


    def save_image(self,file):
        print('compressing and saving data')
        tiff.imwrite(self.filename.removesuffix('.tif')+'_processed_compressed'+'.tif',file,compression=('zlib', 9))
        print('Data compressed and saved')

    def compress_image(self):
        print('attempting data compression')
        try:
            with tiff.TiffFile(self.memap_filename) as tif:
                data = tif.asarray()
                tiff.imwrite(self.filename.removesuffix('.tif')+'_processed_compressed'+'.tif',data,compression=('zlib',9))
                print('Data compressed and saved')
        except:
            print('Data too large for compression, saving uncompressed data instead')
        return                        

if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()

# %%
