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

        self.label = Label(root, text='Upsampleing factor Y:')
        self.label.pack()

        self.upsampling_factor_Y_spinbox = Spinbox(root, from_=1, to=100, width=5)
        self.upsampling_factor_Y_spinbox.set(3)
        self.upsampling_factor_Y_spinbox.pack()
        
        self.label = Label(root, text='Upsampleing factor Z:')
        self.label.pack()

        self.upsampling_factor_Z_spinbox = Spinbox(root, from_=1, to=100, width=5)
        self.upsampling_factor_Z_spinbox.set(3)
        self.upsampling_factor_Z_spinbox.pack()
        

        self.open = Button(root, text='Open Image', command=self.open_image)
        self.open.pack()
        self.process = Button(root, text='Process Image', command=self.upsample)
        self.process.pack()

        self.remapped_image = None
        

    def open_image(self):
        self.filename = filedialog.askopenfilename()
        self.data = np.array(tiff.imread(self.filename))
        self.dim = self.data.ndim
        print('Data shape: '+self.data.shape)

    def upsample(self):
        self.upsampling_factor_Y = int(self.upsampling_factor_Y_spinbox.get())
        self.upsampling_factor_Z = int(self.upsampling_factor_Z_spinbox.get())
        self.remapped_image = np.zeros_like((self.data))
        if self.dim == 2:
            self.process_2D()
        elif self.dim == 3:
            self.process_3D()
        elif self.dim == 4:
            self.process_4D()
        else:
            print('Image dimension not supported')

        self.save_image()
        

    def process_2D(self):
        zoomed_image = sp.ndimage.zoom(self.data,(self.upsampling_factor_Y, 1),order=1)
        self.remapped_image = self.remapping2D(self.remapped_image,zoomed_image)

    def process_3D(self):
        zoomed_image = sp.ndimage.zoom(self.data,(1,self.upsampling_factor_Y, 1),order=1)
        self.remapped_image = self.remapping3D(self.remapped_image,zoomed_image)
    
    def process_4D(self):
        self.zoomed_image = sp.ndimage.zoom(self.data,(1,1,self.upsampling_factor_Y, 1),order=1)
        self.remapped_image = self.remapping4D(self.remapped_image,self.zoomed_image)

    
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
            self.remapped_image[plane] = np.mean(self.zoomed_image[upsampled_plane:upsampled_plane+bins],axis=0)
        
        return remapped_image
    

    def remapping4D(self,remapped_image,zoomed_image):
        # call the correction pipeline for n volumes
        t_dim = remapped_image.shape[0]
        for volume in np.arange(t_dim):
            remapped_image[volume] = self.remapping3D(remapped_image[volume],zoomed_image[volume])
        
        return remapped_image


    def save_image(self):
        if self.remapped_image is not None:
            tiff.imwrite(self.filename.removesuffix('.tif')+'_resampled'+'.tif',self.remapped_image)



if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()

# %%
