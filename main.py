#%%
import os
import glob
from timeit import default_timer as timer  
from time import time 
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
try:
    import tifffile as tiff
    import numpy as np
    from numba import jit, prange
except ImportError:
    print('tifffile, numpy or numba import error, please install with: \n pip install tifffile numpy numba')

from HMF import *

try:
    import rawdata
    import imaging
    import napari_streamin.arrays
    from napari_streamin.read_ird import create_metadata
except ImportError:
    print('napari_streamin import error, can only process tif files')


def timer_func(func): 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print((t2-t1)) #f'Function {func.__name__!r} executed in {(t2-t1):.4f} s'
        return result 
    return wrap_func 

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

##### begin of UI class #####
class App:
    def __init__(self, root):
        # general settings
        self.root = root
        self.root.title('Image Processing')
        
        current_row = 0
        settings_frame = Frame(root)
        settings_frame['borderwidth'] = 2
        settings_frame['relief'] = 'groove'
        settings_frame.grid(row=current_row, column =0,columnspan=2)

        upsampling_values = [2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10]

        Label(settings_frame, text='galvo sin correction:').grid(row=current_row, column=0)
        current_row += 1

        self.do_FDML_correction = BooleanVar(value=True)
        FDML_correction_checkbox = Checkbutton(settings_frame, text='X-FDML 4 buffers correction', variable=self.do_FDML_correction, command=self.X_correction_flipflop_fdml)
        FDML_correction_checkbox.grid(row=current_row, column=1,columnspan=3)

        self.do_x_correction = BooleanVar(value=False)
        do_x_correction_checkbox = Checkbutton(settings_frame, text='X', variable=self.do_x_correction, command=self.X_correction_flipflop_galvo)
        do_x_correction_checkbox.grid(row=current_row, column=0)
        current_row += 1

        self.do_y_correction = BooleanVar(value=True)
        do_y_correction_checkbox = Checkbutton(settings_frame, text='Y', variable=self.do_y_correction)
        do_y_correction_checkbox.grid(row=current_row, column=0)
        current_row += 1

        self.do_z_correction = BooleanVar(value=True)
        do_z_correction_checkbox = Checkbutton(settings_frame, text='Z', variable=self.do_z_correction)
        do_z_correction_checkbox.grid(row=current_row, column=0)
        
        self.rollz = IntVar(value=0)
        Label(settings_frame, text='roll z by:').grid(row=current_row, column=1)
        rollz_spinbox = Spinbox(settings_frame, from_=-100, to=100, width=4, textvariable=self.rollz)
        rollz_spinbox.set(self.rollz.get())
        rollz_spinbox.grid(row=current_row, column=2)
        current_row += 1

        self.flatten4D = BooleanVar(value=False)   
        flatten4D_checkbox = Checkbutton(settings_frame, text='Sum 4D to 2Dt', variable=self.flatten4D)
        flatten4D_checkbox.grid(row=current_row, column=0, columnspan=1)

        Label(settings_frame, text='Upsampleing factor:').grid(row=current_row, column=1, columnspan=2)
        self.upsampling_factor_spinbox = Spinbox(settings_frame, values=upsampling_values, width=4)
        self.upsampling_factor_spinbox.set(upsampling_values[4])
        self.upsampling_factor_spinbox.grid(row=current_row, column=3)
        current_row += 1

        self.snow_threshold_spinbox = Spinbox(settings_frame, from_=0, to=0.99, width=4, increment=0.1, format='%.2f')
        self.snow_threshold_spinbox.set(0.9)
        self.snow_threshold_spinbox.grid(row=current_row, column=3)

        self.remove_snow = BooleanVar(value=False)
        remove_snow_checkbox = Checkbutton(settings_frame, text='remove high value pixels above x*max_value, x =', variable=self.remove_snow)
        remove_snow_checkbox.grid(row=current_row, column=0, columnspan=3)
        current_row += 1

        self.hybrid_median_filter = BooleanVar(value=False)
        apply_hybrid_median_filter_checkbox = Checkbutton(settings_frame, text='hybrid median ', variable=self.hybrid_median_filter)
        apply_hybrid_median_filter_checkbox.grid(row=current_row, column=0)

        self.include_center_pixel = BooleanVar(value=False)
        include_center_pixel_checkbox = Checkbutton(settings_frame, text='center pixel', variable=self.include_center_pixel)
        include_center_pixel_checkbox.grid(row=current_row, column=1)

        Label(settings_frame, text='filter size:').grid(row=current_row, column=2)
        self.filter_size = IntVar(value=3)
        self.filter_size_spinbox = Spinbox(settings_frame, textvariable=self.filter_size, from_=3, to=5, width=4, increment=2)
        self.filter_size_spinbox.grid(row=current_row, column=3)
        current_row += 1

        self.is_single_volume_var = BooleanVar(value=False)
        is_single_volume_checkbox = Checkbutton(settings_frame, text='is single volume', variable=self.is_single_volume_var)
        is_single_volume_checkbox.grid(row=current_row, column=0)

        self.verbose = BooleanVar(value=False)
        verbose = Checkbutton(settings_frame, text='verbose', variable=self.verbose)
        verbose.grid(row=current_row, column=3)

        self.rescale_image = BooleanVar(value=True)
        rescale_image_checkbox = Checkbutton(settings_frame, text='rescale image', variable=self.rescale_image)
        rescale_image_checkbox.grid(row=current_row, column=1)

        # options for time ranges
        current_row = 0
        ranges_frame = Frame(root)
        ranges_frame['borderwidth'] = 2
        ranges_frame['relief'] = 'groove'
        ranges_frame.grid(row=current_row, column =2)

        Label(ranges_frame, text='time ranges (no batch, default all):').grid(row=current_row, column=0, columnspan=3)
        current_row += 1

        #add standard text to entry
        def on_entry_click(event):
            if range_entry.get() == 'start:end':
                range_entry.delete(0, "end") # delete all the text in the entry
                range_entry.insert(0, '') #Insert blank for user input

        def on_focusout(event):
            if range_entry.get() == '':
                range_entry.insert(0, 'start:end')

        # add range on return key
        def on_return(event):
            self.add_range()
        
        
        self.new_range = StringVar()
        range_entry = Entry(ranges_frame, textvariable=self.new_range, width=13)
        range_entry.grid(row=current_row, column=1)
        range_entry.insert(0, 'start:end')
        range_entry.bind('<FocusIn>', on_entry_click)
        range_entry.bind('<FocusOut>', on_focusout)
        range_entry.bind('<Return>', on_return)

        add_button = Button(ranges_frame, text='add range', command=self.add_range)
        add_button.grid(row=current_row, column=2)
        current_row += 1

        remove_last_button = Button(ranges_frame, text='remove last', command=self.remove_last_range)
        remove_last_button.grid(row=current_row, column=2)

        self.text_ranges = Text(ranges_frame, width=10, height=5)
        self.text_ranges.grid(row=current_row, column=1, rowspan=3)
        self.text_ranges.insert(INSERT, 'Ranges:')
        self.text_ranges.config(state=DISABLED)

        # ird settings
        current_row = 0
        ird_frame = Frame(root)
        ird_frame['borderwidth'] = 2
        ird_frame['relief'] = 'groove'
        ird_frame.grid(row=current_row, column =3)
        
        Label(ird_frame, text='ird settings:').grid(row=current_row, column=0, columnspan=3)
        current_row += 1

        self.ird_2d_averaging = IntVar(value=100)
        Label(ird_frame, text='2D averaging:').grid(row=current_row, column=0)
        ird_2d_averaging_spinbox = Spinbox(ird_frame, from_=1, to=100000, width=6, textvariable=self.ird_2d_averaging)
        # ird_2d_averaging_spinbox.set(self.ird_2d_averaging.get())
        ird_2d_averaging_spinbox.grid(row=current_row, column=1)
        current_row += 1

        self.CH2 = BooleanVar(value=False)
        Checkbutton(ird_frame, text='CH 2', variable=self.CH2,  command=self.CH2_toggle).grid(row=current_row, column=0)
        self.both = BooleanVar(value=False)
        Checkbutton(ird_frame, text='both', variable=self.both, command=self.both_toggle).grid(row=current_row, column=1)
        current_row += 1
        Label(ird_frame, text='PSF stack').grid(row=current_row, column=0, columnspan=3)
        current_row += 1
        Button(ird_frame, text='ird folder', command=self.ird_to_tiff_folder).grid(row=current_row, column=0, columnspan=3)
        current_row += 1


        # data buttons
        open_button = Button(root, text='Open Image', command=self.open_image)
        open_button.grid(row=1, column=0, columnspan=1)
        
        process = Button(root, text='Process Image', command=self.process)
        process.grid(row=1, column=1)
    
    def ird_to_tiff_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            files = sorted(glob.glob(os.path.join(folder, '*.ird')))

    def X_correction_flipflop_galvo(self):
        if self.do_x_correction.get() == True:
            self.do_FDML_correction.set(False)
    
    def X_correction_flipflop_fdml(self):
        if self.do_FDML_correction.get() == True:
            self.do_x_correction.set(False)

    def CH2_toggle(self):
        if self.CH2.get() == True:
            self.both.set(False)
    
    def both_toggle(self):
        if self.both.get() == True:
            self.CH2.set(False)
                
    def decide_data_type(self,chanel=0):

        self.is_tiff = False
        self.is_ird = False
        self.is_single_frame = False
        self.is_single_volume = False
        self.is_2D_video = False
        self.is_3D_video = False
        self.channels = 1
        
        try:
            del self.original_t_dim
        except AttributeError:
            pass

        if self.filename.endswith('.ird'):
            self.is_ird = True
            self.ird_file = rawdata.InputFile()
            self.ird_file.open(self.filename)
            self.provider = rawdata.ImageDataProvider(self.ird_file,chanel)
            images = napari_streamin.arrays.VolumeArray(self.provider)
            self.channels = self.ird_file.numChannels()

            if images.shape[0]>1:
                self.is_3D_video = True

            if images.shape[0]==1:
                self.is_single_volume = True

            if images.shape[1]==1:
                images = napari_streamin.arrays.ImageArray(self.provider)
                images._processor.setImageAccumulation(self.ird_2d_averaging.get())
                images._processor.setUndistortion(imaging.ImageGenerator.Undistort__None)

                self.is_2D_video = True
                self.is_3D_video = False
                self.is_single_volume = False

                if images.shape[0]==1:
                    self.is_single_frame = True
                    self.is_2D_video = False

            self.original_t_dim = images.shape[0]
            shape = images.shape
        
        elif self.filename.endswith('.tif') or self.filename.endswith('.tiff'):
            self.is_tiff = True
            with tiff.TiffFile(self.filename) as tif:
                self.dim = tif.series[0].ndim
                self.tif_shape = tif.series[0].shape
                self.dtype = tif.pages[0].dtype
                self.axes = tif.series[0].axes

            if self.dim == 2:
                self.is_single_frame = True
            elif self.dim == 3 and self.is_single_volume_var.get():
                self.is_single_volume = True
            elif self.dim == 3 and not self.is_single_volume_var.get():
                self.is_2D_video = True
                self.original_t_dim = self.tif_shape[0]
            elif self.dim == 4:
                self.is_3D_video = True
                self.original_t_dim = self.tif_shape[0]
            
            else:
                print('Image dimension not supported!')
            shape = self.tif_shape
        return shape

    def open_image(self):
        self.ranges = []
        filenames = filedialog.askopenfilenames(filetypes=[("SLIDE data","*.ird"),("SLIDE data","*.tif"),("SLIDE data","*.tiff")])
        self.filenames = list(filenames)
                   
        for self.filename in self.filenames:
            shape = self.decide_data_type()
            print('Found Stack dimension: '+str(shape)+' in "' + self.filename+'"')
    
    @timer_func
    def process(self):
        self.fdml = self.do_FDML_correction.get()
        self.x_corr = self.do_x_correction.get()
        self.y_corr = self.do_y_correction.get()
        self.z_corr = self.do_z_correction.get()
        self.rescale = self.rescale_image.get()

        self.hmf = self.hybrid_median_filter.get()
        self.fSize = self.filter_size.get()
        self.cPixel = self.include_center_pixel.get()

        self.upsampling_factor = int(self.upsampling_factor_spinbox.get())
        self.melt = self.remove_snow.get()
        self.snow_threshold = float(self.snow_threshold_spinbox.get())

        for self.filename in self.filenames:
            if len(self.filenames)>1: # deactivate ranges for multiple files and reset ranges from previous file
                self.ranges = [] 
            channels = [0]
            # Multi CH support. A new Tiff is generated for each channel due to memmory limitations
            if self.CH2:
                channels = [1]
            if self.both:
                channels = [0,1]
            
            for self.channel in channels:
                self.decide_data_type(self.channel)
                print("Processing: '"+self.filename+"' \nloading data")

                if (self.is_single_frame or self.is_single_volume) and self.is_tiff:
                    with tiff.TiffFile(self.filename) as tif:
                        data = tif.asarray()
                        new_shape = self.create_new_array(data)[0]
                        if self.melt:
                            snow_value = np.amax(data)
                            data = self.melt_snow(data,snow_value)
                        if self.is_single_volume:
                            remapped_image = self.process_3D(data,new_shape)
                        print('processing done')
                        self.save_image(remapped_image)

                elif self.is_3D_video:
                    self.process_4D()
                elif self.is_2D_video or self.is_single_frame:
                    self.process_2Dt()
                else:
                    print('Image dimension not supported!')
                if self.is_ird:
                    self.ird_file.close()
                self.text_ranges.delete(2.0, END)


    def phase_shift_z(self,data,offset):
            return np.roll(data,offset,axis=0)
        
        
    def add_range(self):
        self.text_ranges.config(state=NORMAL)
        self.text_ranges.delete(2.0, END)
        # try:
        if not self.new_range.get() == '':
            a,b=self.new_range.get().replace(' ','').split(':')
            if (a,b) not in self.ranges and int(a) < self.original_t_dim and int(b)<=self.original_t_dim:
                new_start,new_end = a,b
                self.ranges.append((new_start,new_end))
        for start,end in self.ranges:
            self.text_ranges.insert(INSERT, '\n'+start+':'+end)
        self.new_range.set('')
        # except AttributeError:
        #             print('Select a file first')
        self.text_ranges.config(state=DISABLED)

    def remove_last_range(self):
        self.ranges.pop()
        self.add_range()


    def calc_t_dim(self,tif_shape):
        new_t_dim = 0

        for start,end in self.ranges:
            new_t_dim = new_t_dim+int(end)-int(start)

        if new_t_dim > 0:
            t_dim = new_t_dim
            if len(tif_shape) == 3:
                tif_shape = (t_dim,tif_shape[-2],tif_shape[-1])
                z_dim = 1
                if new_t_dim == 1:
                    self.is_single_frame = True
                    self.is_2D_video = False
            else:
                tif_shape = (t_dim,tif_shape[-3],tif_shape[-2],tif_shape[-1])
                z_dim = tif_shape[-3]
        else:
            t_dim = tif_shape[0]
        return t_dim, z_dim, tif_shape
    
        
    def memap(self,shape,name='_TEMP'):
            # create a memmory mapped array to enable processing of larger than RAM files:
            if self.is_tiff:
                memmap_filename = self.filename.replace('.tif',name+'.tif')
            elif self.is_ird:
                memmap_filename = self.filename.replace('.ird',name+'.tif')
            if '_TEMP' in memmap_filename:
                self.in_memmap_filename = memmap_filename
            else:
                self.out_memmap_filename = memmap_filename

            print('Creating memap file, might take a while, shape: '+str(shape))
            try:
                dtype = self.dtype
            except AttributeError:
                dtype = 'uint16'
            # create an empty OME-TIFF file
            start=timer()
            tiff.imwrite(memmap_filename, shape=shape, dtype=dtype, metadata={'axes': self.axes})
            print('Memap file created in ' + str(timer()-start))
        
            # memory map numpy array to data in OME-TIFF file
            memap_stack = tiff.memmap(memmap_filename)
            return memap_stack
    

    def create_new_array(self,data): 
        memmap = False
        if not self.is_2D_video:
            x_dim = data.shape[-1]
            y_dim = data.shape[-2]
            try:
                z_dim = data.shape[-3]
            except IndexError:
                z_dim = 1
            try:
                t_dim = data.shape[-4]
            except IndexError:
                t_dim = 1
        else:
            x_dim = data.shape[-1]
            y_dim = data.shape[-2]
            try:
                t_dim = data.shape[-3]
            except IndexError:
                t_dim = 1
 
        if self.rescale_image.get():
            if self.do_FDML_correction.get():
                x_dim = int(round(x_dim*np.sqrt(2)/(1/2*np.pi)))
            if self.do_x_correction.get():
                x_dim = int(round(x_dim*2/np.pi))
            if self.do_y_correction.get():
                y_dim = int(round(y_dim*2/np.pi))
            if self.do_z_correction.get():
                try:
                    z_dim = int(round(z_dim*2/np.pi))
                except UnboundLocalError:
                    z_dim = 1

        if self.is_single_frame:
            shape = (y_dim,x_dim)
            new_array = np.zeros(shape,dtype='uint16')

        if self.is_single_volume:
            shape = (z_dim,y_dim,x_dim)
            new_array = np.zeros(shape,dtype='uint16')

        if self.is_2D_video:
            shape = (t_dim,y_dim,x_dim)
            memmap, new_array = self.initialize_data_array(shape,name='_processed')
                
        if self.is_3D_video:
            shape = (t_dim,z_dim,y_dim,x_dim)
            memmap, new_array = self.initialize_data_array(shape,name='_processed')
        self.image_out_shape = (t_dim,z_dim,y_dim,x_dim)
        return new_array, memmap


    def load_ird(self, sections, snow_value=0, in_memmap=False, is2Dt=False):
        # import napari_streamin.arrays
        # import imaging # needed to change parameters such as undistort algorithm or sample processor
        match is2Dt:
            case False:
                irdata = napari_streamin.arrays.VolumeArray(self.provider)
                
                self.axes = 'QQYX'
                irdata._processor.setUndistortion(imaging.VolumeGenerator.Undistort__None)

            case True:
                irdata = napari_streamin.arrays.ImageArray(self.provider)
                irdata.image_averaging = self.ird_2d_averaging.get()
                self.axes = 'QYX'
                irdata._processor.setImageAccumulation(self.ird_2d_averaging.get())
                irdata._processor.setUndistortion(imaging.ImageGenerator.Undistort__None)

        tif_shape = irdata.shape
        t_dim, z_dim, tif_shape = self.calc_t_dim(tif_shape)

        in_memmap, data = self.initialize_data_array(tif_shape)

        match is2Dt:
            case False:        
                for index in range(t_dim):
                    data[index] = irdata[sections[index],:,:,:]
                    if self.melt:
                        snow_value = np.maximum(snow_value,np.amax(data[index,:,:,:]))

            case True:
                for index in range(t_dim):
                    data[index] = irdata[sections[index],:,:]
                    if self.melt:
                        snow_value = np.maximum(snow_value,np.amax(data[index,:,:]))
                data = np.squeeze(data)

        return data, t_dim, snow_value, in_memmap

    def initialize_data_array(self, tif_shape, name=''):
        memmap = False
        try:
            data = np.zeros(tif_shape,dtype=np.uint16)
        except np.core._exceptions._ArrayMemoryError:
            memmap = True
            print('MemoryError: File too large for RAM, writing original data to memmap')
            data = self.memap(tif_shape,name=name)
            print('Writing data to memory-mapped array')
        return memmap,data


    def create_section_indices(self):
        if self.ranges == []:
            self.ranges = [(0,self.original_t_dim)]

        # make sections iterable for loop
        for start,end in self.ranges:
            start,end = int(start),int(end)
            try:
                sections = np.append(sections,np.arange(start,end))
            except UnboundLocalError:
                sections = np.arange(start,end)
        return sections  


    def process_4D(self):
        snow_value = 0    

        sections = self.create_section_indices()               

        if self.is_ird:
            data, t_dim, snow_value, in_memmap = self.load_ird(sections)
                            

        elif self.is_tiff:   
            t_dim, z_dim, tif_shape = self.calc_t_dim(self.tif_shape)  
            # Load data either in RAM or as memmap
            in_memmap, data = self.initialize_data_array(tif_shape)

            with tiff.TiffFile(self.filename) as tif:
                for timepoints in range(t_dim):
                    for planes in range(z_dim):
                        data[timepoints,planes] = tif.pages[sections[timepoints]*z_dim+planes].asarray()
                        if self.melt:
                            snow_value = np.maximum(snow_value,np.amax(data[timepoints,planes]))
                    if timepoints % 50 == 0:
                        print('loading '+ str(timepoints) + '/' + str(t_dim) + ' volumes')
        print('Data loaded')

        offset = self.rollz.get()
        if offset != 0:
            #roll every 2nd volume
            for timestep in range(t_dim):
                if timestep % 2 == 0:
                    data[timestep] = self.phase_shift_z(data[timestep], offset)

        # melt snow if selected
        if self.melt:
            for timestep in range(t_dim):
                data[timestep] = self.melt_snow(data[timestep],snow_value)
                if timestep % 50 == 0:
                    print('removed snow in '+str(timestep)+' Volumes')
        if self.rescale_image.get():
            print('Creating tif with corrected aspect ratio')
        new_shape,out_memmap = self.create_new_array(data)

        # process data
        print('filtering data')
        if self.do_z_correction.get() or self.do_y_correction.get() or self.do_x_correction.get() or self.do_FDML_correction.get() or self.hybrid_median_filter.get():
            start=timer()
            for timestep in range(t_dim):
                new_shape[timestep] = self.process_3D(data[timestep],new_shape[0])
                if timestep % 20 == 0:
                    print(str(timestep)+' volumes corrected')
                    if self.verbose.get():
                        print('Time elapsed: '+str(timer()-start))
                        start=timer()
        if self.is_ird:
                self.ird_file.close()
        if in_memmap:
            data.flush()

        if self.flatten4D.get():
            print('Flattening 4D data')
            new_shape = flatten_4D(new_shape)
            data = flatten_4D(data)
        
        self.save_data(data,new_shape,in_memmap,out_memmap) 


    def process_2Dt(self):
        snow_value = 0

        sections = self.create_section_indices()

        if self.is_ird:
            data, t_dim, snow_value, in_memmap = self.load_ird(sections,is2Dt=True)
            
        elif self.is_tiff:
            t_dim, z_dim, tif_shape = self.calc_t_dim(self.tif_shape)  
            # Load data either in RAM or as memmap
            in_memmap, data = self.initialize_data_array(tif_shape)

            with tiff.TiffFile(self.filename) as tif:
                for timepoints in range(t_dim):
                    data[timepoints] = tif.pages[sections[timepoints]].asarray()
                    if self.melt:
                        snow_value = np.maximum(snow_value,np.amax(data[timepoints]))
                    if timepoints % 50 == 0:
                        print('loading '+ str(timepoints) + '/' + str(t_dim) + ' volumes')
        print('Data loaded')
            
        # melt snow 2D if selected
        if self.melt:
            print('Max Snow value: '+str(snow_value) + ' filtering all values above ' + str(self.snow_threshold*snow_value))
            for timestep in np.arange(t_dim): 
                data[timestep] = self.melt_snow(data[timestep],snow_value)
            print('Snow removed')

        # create new array with corrected aspect ratio
        new_shape,out_memmap = self.create_new_array(data)

        if self.is_2D_video:
            for timestep in np.arange(t_dim): 
                new_shape[timestep] = self.process_2D(data[timestep],new_shape[0])
                if timestep % 50 == 0:
                    print('Frame '+str(timestep)+' corrected')
        elif self.is_single_frame:
            new_shape = self.process_2D(data,new_shape)
            

        self.save_data(data,new_shape,in_memmap,out_memmap)          
        return


    def process_3D(self,data,shape_array):

        if self.x_corr or self.fdml:
            remapped_data = np.zeros((data.shape[0],data.shape[1],shape_array.shape[2]),dtype = 'uint16')
            remapped_data = np.swapaxes(remapped_data,1,2)
            shape_array = np.swapaxes(shape_array,1,2)
            data = np.swapaxes(data,1,2)

            remapped_data = remapping3D(data,shape_array,self.upsampling_factor,self.fdml)
            
            shape_array = np.swapaxes(shape_array,1,2)
            data = remapped_data
            data = np.swapaxes(data,1,2)

        if self.y_corr:
            remapped_data = np.zeros((data.shape[0],shape_array.shape[1],data.shape[2]),dtype = 'uint16')
            remapped_data = remapping3D(data,shape_array,self.upsampling_factor)
            data = remapped_data

        if self.z_corr:
            remapped_data = np.zeros((shape_array.shape[0],data.shape[1],data.shape[2]),dtype = 'uint16')
            shape_array = np.swapaxes(shape_array,0,1)
            remapped_data = np.swapaxes(remapped_data,0,1)
            data = np.swapaxes(data,0,1)
            
            remapped_data = remapping3D(data,shape_array,self.upsampling_factor)
            
            data=remapped_data
            data = np.swapaxes(data,0,1)

        if self.hmf:
            data = hybrid_3d_median_filter(data,include_center_pixel=self.cPixel)
        return data

    def process_2D(self,data,shape_array):


        if not self.x_corr and not self.fdml and not self.y_corr:
            remapped_image = data

        if self.y_corr:
            remapped_image = remapping2D(data,shape_array,self.upsampling_factor)
            data=remapped_image

        if self.x_corr or self.fdml:
            shape_array = np.swapaxes(shape_array,0,1)
            remapped_image = np.swapaxes(data,0,1)
            remapped_image = remapping2D(remapped_image,shape_array,self.upsampling_factor,self.fdml)
            remapped_image = np.swapaxes(remapped_image,0,1)

        if self.hmf:
            remapped_image = hybrid_2d_median_filter(remapped_image,include_center_pixel=self.cPixel,filtersize=self.fSize)
        return remapped_image

### Snow removal ###

    def melt_snow(self,data,snow_value):
        if self.verbose.get():
            print('Max Snow value: '+str(snow_value))
            print('Remvoing snow above '+str(self.snow_threshold*snow_value))
        filtered_data = data
        snow_coords = list(zip(*np.where(data > self.snow_threshold*snow_value)))
        x_dim = data.shape[-1]
        extended_coords=[]
        if data.ndim == 2:
            data_shape = (data.shape[-2]-1,data.shape[-1]-1)
            kernel = np.ones((3,3))/6
            kernel[1,:]=0

            # extend snow coordinates to include neighbouring pixels in x direction
            for flakes in snow_coords:
               
                if flakes[-1]-1 < 0:
                    pass
                else:
                    extended_coords.append([flakes[-2],flakes[-1]-1])

                extended_coords.append([flakes[-2],flakes[-1]])
                
                if flakes[-1]+1 > x_dim-2:
                    pass
                else:
                    extended_coords.append([flakes[-2],flakes[-1]+1])   

                if flakes[-1]+2 > x_dim-3:
                    pass
                else:
                    extended_coords.append([flakes[-2],flakes[-1]+2])

            # remove duplicates
            new_snow_coords = list(set(map(tuple,extended_coords)))

            # filter snow
            for flakes in new_snow_coords:
                if all(a < b for a, b in zip(flakes,data_shape)) and all(a > b for a, b in zip(flakes,(0,0))):
                    filtered_data[flakes] = np.round(np.sum((data[flakes[-2]-1:flakes[-2]+2,flakes[-1]-1:flakes[-1]+2]*kernel)*0.01)*100).astype('uint16')
                else:
                    filtered_data[flakes] = 0

        elif data.ndim == 3:
            data_shape = (data.shape[-3]-1,data.shape[-2]-1,data.shape[-1]-1)
            kernel = np.ones((3,3,3))/24
            kernel[1,1,:]=0

            # extend snow coordinates to include neighbouring pixels in x direction
            for flakes in snow_coords:
                if flakes[-1]-1 < 0:
                    pass
                else:
                    extended_coords.append([flakes[-3],flakes[-2],flakes[-1]-1])

                extended_coords.append([flakes[-3],flakes[-2],flakes[-1]])

                if flakes[-1]+1 > x_dim-2:
                    pass
                else:
                    extended_coords.append([flakes[-3],flakes[-2],flakes[-1]+1])
                if flakes[-1]+2 > x_dim-3:
                    pass
                else:
                    extended_coords.append([flakes[-3],flakes[-2],flakes[-1]+2])

            # remove duplicates
            new_snow_coords = list(set(map(tuple,extended_coords)))

            # filter snow
            for flakes in new_snow_coords:
                if all(a < b for a, b in zip(flakes,data_shape)) and all(a > b for a, b in zip(flakes,(0,0,0))):
                    filtered_data[flakes] = np.round(np.sum(data[flakes[-3]-1:flakes[-3]+2,flakes[-2]-1:flakes[-2]+2,flakes[-1]-1:flakes[-1]+2]*kernel)).astype('uint16')
                # else:
                    filtered_data[flakes] = 0

        return filtered_data

    def make_metadata(self):
        if not self.is_ird:
            return {'axes': self.get_axes()}
        
        averaging = 1 
        meta3D = {}
        ird_metadata = create_metadata(self.ird_file)
        px_size_x =round(float(ird_metadata['MM/Machine/ScaleX'])*float(ird_metadata['MM/Laser/SweepRange'])/self.image_out_shape[-1],2)
        px_size_y =round(float(ird_metadata['MM/Machine/ScaleY'])*float(ird_metadata['MM/FunX/SineAmplitude'])/self.image_out_shape[-2],2)
        px_size_z =round(float(ird_metadata['MM/Machine/ScaleZ'])*float(ird_metadata['MM/FunY/SineAmplitude'])/self.image_out_shape[-3],2)
        

        if self.is_2D_video:
            averaging = self.ird_2d_averaging.get()
            time_increment = 1/(float(ird_metadata['MM/Laser/SweepFrequency'])*1e6/(float(ird_metadata['MM/FunX/SineLength'])*2))
        elif self.is_3D_video:
            time_increment = 1/(float(ird_metadata['MM/Laser/SweepFrequency'])*1e6/(float(ird_metadata['MM/FunX/SineLength'])*2*float(ird_metadata['MM/FunY/SineLength'])))
            meta3D = {
                'PhysicalSizeZ': px_size_z,
                'PhysicalSizeZUnit': 'µm',
                'spacing': px_size_z,
                }
        else:
            time_increment = 1

        metadata = {
            'axes': self.get_axes(),
            'unit': 'µm',
            'TimeIncrement': time_increment*averaging,
            'TimeIncrementUnit': 's',
            'PhysicalSizeX': px_size_x,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': px_size_y,
            'PhysicalSizeYUnit': 'µm',
            'unit': 'µm',
            'fps': 1/(time_increment*averaging)
        }
        metadata.update(meta3D)
        return metadata
    
    def save_data(self,data,new_shape,in_memmap,out_memmap):
        print('Saving data')
        if not np.any(new_shape):
            if in_memmap:
                data.flush()
                try:
                    path=self.in_memmap_filename.replace('_TEMP','_processed')
                except:
                    pass
                try:
                    os.rename(self.in_memmap_filename,path)
                    self.compress_image(path) 
                except:
                    self.compress_image(self.in_memmap_filename) 
            else:    
                self.save_image(data)
        else:
            if in_memmap:
                try:
                    data.flush()
                    os.remove(self.in_memmap_filename)
                except:
                    pass
            if out_memmap:
                try:
                    new_shape.flush()
                    self.compress_image(self.out_memmap_filename)
                except:
                    pass
            else:
                self.save_image(new_shape)     
        return
    
    def get_axes(self):
        if self.is_single_frame:
            axes = 'YX'
        elif self.is_2D_video:
            axes = 'TYX'
        elif self.is_single_volume:
            axes = 'TYX'
        else:
            axes = 'TZYX'
        if self.flatten4D.get():
            axes = 'TYX'
        return axes
    
    def set_filename(self):
        
        processing = ([self.fdml,self.x_corr,self.y_corr,self.z_corr,self.rescale])

        modstring = ''

        if self.CH2 or self.both:
            modstring = modstring + f'_CH{self.channel+1}'

        if self.is_ird and (self.is_single_frame or self.is_2D_video):
            modstring = modstring+f'_{str(self.ird_2d_averaging.get())}x_avg'

        if self.hmf:
            if self.is_single_frame or self.is_2D_video:
                if self.cPixel:
                    modstring=modstring+f'_hmf-{self.fSize}x{self.fSize}+'
                else:
                    modstring=modstring+f'_hmf-{self.fSize}x{self.fSize}-'
            else:
                if self.cPixel:
                    modstring=modstring+f'_hmf+'
                else:
                    modstring=modstring+f'_hmf-'

        if self.rollz.get() != 0:
            modstring = modstring + f'_rollz_{self.rollz.get()}'

        filename_out = self.filename.replace('.ome','').replace('.tif', '_processed.ome.tif').replace('.ird', '_processed.ome.tif')
        if not any(processing):
            filename_out = filename_out.replace(f'_processed.ome.tif',f'{modstring}.ome.tif')
        else:
            filename_out = filename_out.replace(f'.ome.tif',f'{modstring}.ome.tif')
        return filename_out
    

    def save_image(self,data):
        print('compressing and saving data')
        axes = self.get_axes()
        self.set_filename()
        tiff.imwrite(
            self.set_filename(),            
            data,
            ome=TRUE,
            bigtiff=TRUE,
            photometric='minisblack',
            compression='zlib',
            compressionargs={'level': 6},
            metadata=self.make_metadata()
            )
        del data
        print('Data compressed and saved')

    def compress_image(self,path):
        print('attempting data compression')
        try:
            with tiff.TiffFile(path) as tif:
                data = tif.asarray()
                axes = self.get_axes()
                
                tiff.imwrite(self.set_filename(),
                    data,
                    ome=TRUE,
                    bigtiff=TRUE,
                    photometric='minisblack',
                    compression='zlib',
                    compressionargs={'level': 8},
                    metadata=self.make_metadata()

                    )
            print('Data compressed and saved')
        except np.core._exceptions._ArrayMemoryError:
            print('Data too large for RAM, saved uncompressed data instead')
        return                        

if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()
