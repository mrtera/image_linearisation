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
        self.label.grid(row=0, column=0)
        self.upsampling_factor_X_spinbox = Spinbox(root, from_=1, to=100, width=3)
        self.upsampling_factor_X_spinbox.set(3)
        self.upsampling_factor_X_spinbox.grid(row=0, column=1)

        self.label = Label(root, text='Upsampleing factor Y:')
        self.label.grid(row=1, column=0)
        self.upsampling_factor_Y_spinbox = Spinbox(root, from_=1, to=100, width=3)
        self.upsampling_factor_Y_spinbox.set(3)
        self.upsampling_factor_Y_spinbox.grid(row=1, column=1)

        self.label = Label(root, text='Upsampleing factor Z:')
        self.label.grid(row=2, column=0)
        self.upsampling_factor_Z_spinbox = Spinbox(root, from_=1, to=100, width=3)
        self.upsampling_factor_Z_spinbox.set(3)
        self.upsampling_factor_Z_spinbox.grid(row=2, column=1)

        self.remove_snow = BooleanVar(value=True)
        self.remove_snow_checkbox = Checkbutton(root, text='removev snow', variable=self.remove_snow)
        self.remove_snow_checkbox.grid(row=4, column=0)

        self.is2D_video = BooleanVar(value=False)
        self.is2D_video_checkbox = Checkbutton(root, text='Is 2D Video', variable=self.is2D_video)
        self.is2D_video_checkbox.grid(row=3, column=1)

        self.do_x_correction = BooleanVar(value=False)
        self.do_x_correction_checkbox = Checkbutton(root, text='Do X correction', variable=self.do_x_correction)
        self.do_x_correction_checkbox.grid(row=3, column=0)

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
            self.data = tif.asarray()
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
        self.upsampling_factor_Y = int(self.upsampling_factor_Y_spinbox.get())
        self.upsampling_factor_Z = int(self.upsampling_factor_Z_spinbox.get())
        self.is2D = self.is2D_video.get()
        self.melt = self.remove_snow.get()
        if self.dim >=2 and self.dim <= 3 and self.is2D == False:

            with tiff.TiffFile(self.filename) as tif:
                self.data = tif.asarray()
            self.remapped_image = np.zeros_like((self.data))
            if self.dim == 2:
                self.process_2D()
                self.save_image()
            elif self.dim == 3:
                self.process_3D()
                self.save_image()

        elif self.dim == 4 and self.is2D == False:
                self.process_4D()

        elif self.dim == 3 and self.is2D == True:
            ...

        else:
            print('Image dimension not supported!')
        

    def memap(self):
            # create a memmory mapped array to enable processing of larger than RAM files:
            filename = self.filename.removesuffix('.tif')+'_processed'+'.tif'
            shape = self.tif_shape
            print('memap file shape: '+str(shape))
            dtype = self.dtype
            # create an empty OME-TIFF file
            tiff.imwrite(filename, shape=shape, dtype=dtype, metadata={'axes': self.axes})

            # memory map numpy array to data in OME-TIFF file
            memap_stack = tiff.memmap(filename)
            return memap_stack
           
        
    #### X correction
    # if self.do_x_correction.get() == True:
    #             self.data = np.swapaxes(self.data,1,2)
    #         self.remapped_image = np.zeros_like((self.data))
    #         self.process_2D()
    #         self.save_image()
        

    def process_2D(self):
        zoomed_image = sp.ndimage.zoom(self.data,(self.upsampling_factor_Y, 1),order=1)
        self.remapped_image = self.remapping2D(self.remapped_image,zoomed_image)

    def process_3D(self):
        zoomed_image = sp.ndimage.zoom(self.data,(1,self.upsampling_factor_Y, 1),order=1)
        self.remapped_image = self.remapping3D(self.remapped_image,zoomed_image)
    
    def process_4D(self):
        memap_stack = self.memap()
        # write data to memory-mapped array
        with tiff.TiffFile(self.filename) as tif:
            for timepoints in range(self.t_dim):
                for volumes in range(self.z_dim):
                    memap_stack[timepoints,volumes] = tif.pages[timepoints*self.z_dim+volumes].asarray()
        print('Data written to memory-mapped array') 
        # process data in memory-mapped array
        
        for timestep in np.arange(self.t_dim): 
            zoomed_image = sp.ndimage.zoom(memap_stack[timestep],(1,self.upsampling_factor_Y, 1),order=1)
            memap_stack[timestep] = self.remapping3D(memap_stack[timestep],zoomed_image)
            print('Volume '+str(timestep)+' corrected')
        memap_stack.flush()

    
    def correction_factor(self,current_index, max_index):
        return 1/(np.pi*np.sqrt(-1*(current_index+1/2)*(current_index+1/2-max_index)))


    def remapping2D(self,remapped_image,zoomed_image):
        sum_correction_factor = 0
        y_dim=remapped_image.shape[0]
        y_dim_upsampled = zoomed_image.shape[0]
        for row in np.arange(y_dim):
            correction_factor = self.correction_factor(row,y_dim)
            sum_correction_factor += correction_factor
            upsampled_row = np.round(y_dim_upsampled*sum_correction_factor).astype(int)
            bins= np.round(y_dim*self.upsampling_factor_Y*correction_factor).astype(int)
            remapped_image[row] = np.mean(zoomed_image[upsampled_row:upsampled_row+bins],axis=0)
        
        return remapped_image

    
    def remapping3D(self,remapped_image,zoomed_image):  

        # correct all slices in Y 
        z_dim=remapped_image.shape[0]
        for plane in np.arange(z_dim):
            remapped_image[plane] = self.remapping2D(remapped_image[plane],zoomed_image[plane])
   
        # correct Volume in Z
        sum_correction_factor_Z = 0
        zoomed_image = sp.ndimage.zoom(remapped_image,(self.upsampling_factor_Z, 1, 1),order=1)
        z_dim_upsampled = zoomed_image.shape[0]

        for plane in np.arange(z_dim):
            correction_factor_Z = self.correction_factor(plane,z_dim)
            sum_correction_factor_Z += correction_factor_Z
            upsampled_plane = np.round(z_dim_upsampled*sum_correction_factor_Z).astype(int)
            bins= np.round(z_dim*self.upsampling_factor_Z*correction_factor_Z).astype(int)
            remapped_image[plane] = np.mean(zoomed_image[upsampled_plane:upsampled_plane+bins],axis=0)
        
        return remapped_image
    

    def save_image(self):
        if self.remapped_image is not None:
            tiff.imwrite(self.filename.removesuffix('.tif')+'_processed'+'.tif',self.remapped_image,compression=('zlib', 1))

if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()

# %%
