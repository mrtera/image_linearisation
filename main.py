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
        print(self.data.shape)

    def upsample(self):
        self.upsampling_factor_Y = int(self.upsampling_factor_Y_spinbox.get())
        self.upsampling_factor_Z = int(self.upsampling_factor_Z_spinbox.get())
        self.remapped_image = np.zeros_like((self.data))
        if self.dim == 2:
            self.upsample_2D()
        elif self.dim == 3:
            self.upsample_3D()
        elif self.dim == 4:
            self.upsample_4D()
        else:
            print('Image dimension not supported')
        self.save_image()
        

    def upsample_2D(self):
        self.zoomed_image = sp.ndimage.zoom(self.data,(self.upsampling_factor_Y, 1),order=1)
        self.remapping2D()

    def upsample_3D(self):
        self.zoomed_image = sp.ndimage.zoom(self.data,(1,self.upsampling_factor_Y, 1),order=1)
        self.remapping3D2()
    
    def upsample_4D(self):
        self.zoomed_image = sp.ndimage.zoom(self.data,(1,1,self.upsampling_factor_Y, 1),order=1)
        self.remapping4D()

    
    def correction_factor(self,current_index, max_index):
        return 1/(np.pi*np.sqrt(-1*(current_index+1/2)*(current_index+1/2-max_index)))


    def remapping2D(self):
        sum_correction_factor = 0
        y_dim=self.remapped_image.shape[0]
        y_dim_upsampled = self.zoomed_image.shape[0]
        for row in np.arange(y_dim):
            correction_factor = self.correction_factor(row,y_dim)
            sum_correction_factor += correction_factor
            upsampled_row = np.round(y_dim_upsampled*sum_correction_factor).astype(int)
            bins= np.round(y_dim*self.upsampling_factor_Y*correction_factor).astype(int)
            self.remapped_image[row] = np.mean(self.zoomed_image[upsampled_row:upsampled_row+bins],axis=0)

    
    def remapping3D2(self):  
        sum_correction_factor_Z = 0
        y_dim=self.remapped_image.shape[1]
        y_dim_upsampled = self.zoomed_image.shape[1]
        z_dim=self.remapped_image.shape[0]

        # correct slices in Y 
        for plane in np.arange(z_dim):
            sum_correction_factor_Y = 0
            for row in np.arange(y_dim):
                correction_factor_Y = self.correction_factor(row,y_dim)
                sum_correction_factor_Y += correction_factor_Y
                upsampled_row = np.round(y_dim_upsampled*sum_correction_factor_Y).astype(int)
                bins= np.round(y_dim*self.upsampling_factor_Y*correction_factor_Y).astype(int)
                self.remapped_image[plane,row,:] = np.mean(self.zoomed_image[plane,upsampled_row:upsampled_row+bins,:],axis=0)
        
        # correct Volume in Z
        
        self.zoomed_image = sp.ndimage.zoom(self.remapped_image,(self.upsampling_factor_Z, 1, 1),order=1)
        z_dim_upsampled = self.zoomed_image.shape[0]

        for plane in np.arange(z_dim):
            correction_factor_Z = self.correction_factor(plane,z_dim)
            sum_correction_factor_Z += correction_factor_Z
            upsampled_plane = np.round(z_dim_upsampled*sum_correction_factor_Z).astype(int)
            bins= np.round(z_dim*self.upsampling_factor_Z*correction_factor_Z).astype(int)
            self.remapped_image[plane] = np.mean(self.zoomed_image[upsampled_plane:upsampled_plane+bins],axis=0)


    def save_image(self):
        if self.remapped_image is not None:
            tiff.imwrite(self.filename.removesuffix('.tif')+'_resampled'+'.tif',self.remapped_image)



if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()

# %%
