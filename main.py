#%%
from PIL import Image, ImageTk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import tifffile as tiff
import numpy as np
import scipy as sp

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

        self.snow_threshold_spinbox = Spinbox(root, from_=0, to=0.99, width=4, increment=0.01, format='%.2f')
        self.snow_threshold_spinbox.set(0.1)
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

        self.open = Button(root, text='Open Image', command=self.open_image)
        self.open.grid(row=5, column=0, columnspan=1)
        self.process = Button(root, text='Process Image', command=self.upsample)
        self.process.grid(row=5, column=1,)

        self.remapped_image = None

    def open_image(self):

        self.filename = filedialog.askopenfilename()
        with tiff.TiffFile(self.filename) as tif:
            self.dim = tif.series[0].ndim
            self.tif_shape = tif.series[0].shape
            self.dtype = tif.pages[0].dtype
            self.axes = tif.series[0].axes
            if self.dim >=2 and self.dim <= 4 and self.is2D_video.get() == False:
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
            elif self.dim == 3 and self.is2D_video.get() == True:
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
        if self.dim >=2 and self.dim <= 3 and self.is2D == False:

            with tiff.TiffFile(self.filename) as tif:
                data = tif.asarray()
                if self.melt == True:
                    snow_value = np.amax(data)
            if self.dim == 2:
                if self.melt == True:
                    data = self.melt_snow(data,snow_value,D2=True)
                remapped_image = self.process_2D(data)
                self.save_image(remapped_image)
            elif self.dim == 3:
                if self.melt == True:
                    data = self.melt_snow(data,snow_value)
                remapped_image = self.process_3D(data)
                self.save_image(remapped_image)

        elif self.dim == 4 and self.is2D == False:
            remapped_image = self.process_4D()
            print('Data saved')

        elif self.dim == 3 and self.is2D == True:
            remapped_image = self.process_2Dt()
            print('Data saved')

        else:
            print('Image dimension not supported!')
        

    def memap(self):
            # create a memmory mapped array to enable processing of larger than RAM files:
            filename = self.filename.removesuffix('.tif')+'_processed'+'.tif'
            shape = self.tif_shape
            print('Creating memap file, might take a while, shape: '+str(shape))
            dtype = self.dtype
            # create an empty OME-TIFF file
            tiff.imwrite(filename, shape=shape, dtype=dtype, metadata={'axes': self.axes})

            # memory map numpy array to data in OME-TIFF file
            memap_stack = tiff.memmap(filename)
            return memap_stack
    
    
    def process_4D(self):
        memap_stack = self.memap()
        # write data to memory-mapped array
        print('Writing data to memory-mapped array')
        with tiff.TiffFile(self.filename) as tif:
            for timepoints in range(self.t_dim):
                for volumes in range(self.z_dim):
                    memap_stack[timepoints,volumes] = tif.pages[timepoints*self.z_dim+volumes].asarray()
                if timepoints % 10 == 0:
                    print(str(timepoints) + '/' + str(self.t_dim) + ' Volumes written')
        print('Data written to memory-mapped array') 
        
        # process data in memory-mapped array
        # melt snow if selected
        if self.melt == True:
            snow_value = np.amax(memap_stack)
            print('Max Snow value: '+str(snow_value) + ' filtering all values above ' + str(int(self.snow_threshold*snow_value)))
            for timestep in range(self.t_dim):
                memap_stack[timestep] = self.melt_snow(memap_stack[timestep],snow_value)
                zoomed_image = sp.ndimage.zoom(memap_stack[timestep],(1,self.upsampling_factor_Y, 1),order=1)
                memap_stack[timestep] = self.remapping3D(memap_stack[timestep],zoomed_image)
                print('Volume '+str(timestep)+' corrected')
        
        else:
            for timestep in np.arange(self.t_dim): 
                zoomed_image = sp.ndimage.zoom(memap_stack[timestep],(1,self.upsampling_factor_Y, 1),order=1)
                memap_stack[timestep] = self.remapping3D(memap_stack[timestep],zoomed_image)
                print('Volume '+str(timestep)+' corrected')
        memap_stack.flush()        
        
        return memap_stack
    
    
    def process_2Dt(self):
        memap_stack = self.memap()
        # write data to memory-mapped array
        print('Writing data to memory-mapped array')
        with tiff.TiffFile(self.filename) as tif:
            for timepoints in range(self.z_dim):
                if timepoints % 100 == 0:
                    print(str(timepoints) + '/' + str(self.z_dim) + ' Frames written')
                memap_stack[timepoints] = tif.pages[timepoints].asarray()
        print('Data written to memory-mapped array')
        
        # process data in memory-mapped array
        # melt snow 2D if selected
        if self.melt == True:
            snow_value = np.amax(memap_stack)
            print('Max Snow value: '+str(snow_value) + ' filtering all values above ' + str(self.snow_threshold*snow_value))
            for timestep in np.arange(self.z_dim): 
                memap_stack[timestep] = self.melt_snow(memap_stack[timestep],snow_value,D2=True)
                memap_stack[timestep] = self.process_2D(memap_stack[timestep])
                print('Frame '+str(timestep)+' corrected')
            self.save_image(memap_stack)
        
        else:
            for timestep in np.arange(self.z_dim): 
                memap_stack[timestep] = self.melt_snow(memap_stack[timestep],snow_value,D2=True)
                memap_stack[timestep] = self.process_2D(memap_stack[timestep])
                print('Frame '+str(timestep)+' corrected')
            self.save_image(memap_stack)

        memap_stack.flush()        
        return memap_stack
    
    
    def process_3D(self,data):
        zoomed_image = sp.ndimage.zoom(data,(1,self.upsampling_factor_Y, 1),order=1)
        remapped_image = self.remapping3D(data,zoomed_image)
        return remapped_image
                   

    def process_2D(self,data):
        remapped_image = self.remapping2D(data)
        return remapped_image


### Remapping ###
    def remapping3D(self,remapped_image,zoomed_image): 
        
        # correct all slices in Y 
        z_dim=remapped_image.shape[0]
        for plane in np.arange(z_dim):
            remapped_image[plane] = self.remapping2D(remapped_image[plane])

        # correct Volume in Z
        if self.do_z_correction.get() == True:
            sum_correction_factor_Z = 0
            zoomed_image = sp.ndimage.zoom(remapped_image,(self.upsampling_factor_Z, 1, 1),order=1)
            z_dim_upsampled = zoomed_image.shape[0]

            for plane in np.arange(z_dim):
                correction_factor_Z = self.correction_factor(plane,z_dim)
                sum_correction_factor_Z += correction_factor_Z
                upsampled_plane = np.round(z_dim_upsampled*sum_correction_factor_Z).astype(int)
                bins= np.round(z_dim*self.upsampling_factor_Z*correction_factor_Z).astype(int)
                remapped_image[plane] = np.mean(zoomed_image[upsampled_plane:upsampled_plane+bins],axis=0)
        else:
            pass
        
        return remapped_image


    def remapping2D(self,remapped_image):
        if self.do_Y_correction.get() == True:
            zoomed_image = sp.ndimage.zoom(remapped_image,(self.upsampling_factor_Y, 1),order=1)
            remapped_image = self.remapping1D(remapped_image,zoomed_image,self.upsampling_factor_Y)        
        if self.do_x_correction.get() == True:
            remapped_image = np.swapaxes(remapped_image,0,1)
            zoomed_image = sp.ndimage.zoom(remapped_image,(self.upsampling_factor_X, 1),order=1)
            remapped_image = self.remapping1D(remapped_image,zoomed_image,self.upsampling_factor_X)
            remapped_image = np.swapaxes(remapped_image,0,1)  

        return remapped_image    
        
    
    def remapping1D(self,remapped_image,zoomed_image,upsampling_factor):
        sum_correction_factor = 0
        dim=remapped_image.shape[0]
        dim_upsampled = zoomed_image.shape[0]
        for row in np.arange(dim):
            correction_factor = self.correction_factor(row,dim)
            sum_correction_factor += correction_factor
            upsampled_row = np.round(dim_upsampled*sum_correction_factor).astype(int)
            bins= np.round(dim*upsampling_factor*correction_factor).astype(int)
            remapped_image[row] = np.mean(zoomed_image[upsampled_row:upsampled_row+bins],axis=0)
        return remapped_image
    

    def correction_factor(self,current_index, max_index):
        return 1/(np.pi*np.sqrt(-1*(current_index+1/2)*(current_index+1/2-max_index)))

### Snow removal ###
    def melt_snow(self,data,snow_value,D2=False):
        filtered_data = data
        if D2 == True:
            snow_coords = list(zip(*np.where(data > self.snow_threshold*snow_value)))
            for flakes in snow_coords:
                try:
                    filtered_data[flakes] = np.mean(data[flakes[0]-1:flakes[0]+2:2,flakes[1]-1:flakes[1]+2]).astype('uint16')
                except IndexError:
                    filtered_data[flakes] = 0
                except RuntimeWarning:
                    pass

        else:
            snow_coords = list(zip(*np.where(data > self.snow_threshold*snow_value)))
            for flakes in snow_coords:
                try:
                    filtered_data[flakes] = np.mean(data[flakes[0]-1:flakes[0]+2,flakes[1]-1:flakes[1]+2,flakes[2]-1:flakes[2]+2:2]).astype('uint16')
                except IndexError:
                    filtered_data[flakes] = 0
                except RuntimeWarning:
                    pass

        return data


    def save_image(self,file):
        tiff.imwrite(self.filename.removesuffix('.tif')+'_processed'+'.tif',file,compression=('zlib', 1))
        print('Data saved')

if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()

# %%
