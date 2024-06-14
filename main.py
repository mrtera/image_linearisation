#%%
from PIL import Image, ImageTk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import tifffile as tiff
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import imageio as io
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Processing')
        self.root.resizable(True, True)

        self.label = Label(root, text='Image Linearisation')
        self.label.pack()

        self.button = Button(root, text='Open Image', command=self.open_image)
        self.button.pack()

        self.canvas = Canvas(root, width=600, height=400)
        self.canvas.pack()
        

    def open_image(self):
        self.filename = filedialog.askopenfilename()
        self.data = np.array(tiff.imread(self.filename))
        print(self.data.shape)
        self.upsample()


    def upsample(self):
        self.upsampling_factor=16
        self.zoomed_image = sp.ndimage.zoom(self.data,(self.upsampling_factor, 1),order=1)
        self.remapping()

    def remapping(self):
        self.remapped_image = np.zeros((self.data.shape[0],self.data.shape[1]))
        sum_correction_factor = 0
        y_dim=self.remapped_image.shape[0]
        y_dim_upsampled = self.zoomed_image.shape[0]
        for row in np.arange(y_dim):
            correction_factor = 1/(np.pi*np.sqrt(-1*(row+1/2)*(row+1/2-y_dim)))
            sum_correction_factor += correction_factor
            upsampled_row = np.round(y_dim_upsampled*sum_correction_factor).astype(int)
            bins= np.round(y_dim*self.upsampling_factor*correction_factor).astype(int)
            self.remapped_image[row] = np.mean(self.zoomed_image[upsampled_row:upsampled_row+bins],axis=0)
        self.save_image()
    
    def save_image(self):
        tiff.imsave('remapped_image.tif',self.remapped_image)



if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()
