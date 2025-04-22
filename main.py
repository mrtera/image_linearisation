#%%
import os
from PIL import Image, ImageTk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import tifffile as tiff
import numpy as np
import scipy as sp
from numba import jit, prange
from timeit import default_timer as timer  
from time import time 
try:
    import rawdata
    import napari_streamin.arrays
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
# @timer_func
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
                index = (row+dim/2)/2+1/2
                max_dim =3/4*dim
                correction_factor = 1/(np.pi*np.sqrt(1.5*dim-row)*np.sqrt(row+1/2*dim)) 
                # correction_factor = 1/(np.pi*np.sqrt(-1*index*(index-max_dim)))
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
            index = (row+dim+1/2)/2
            correction_factor = 1/(np.pi*np.sqrt(-1*index*(index-dim)))
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

        label = Label(settings_frame, text='galvo sin correction:')
        label.grid(row=current_row, column=0)
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
        current_row += 1     

        self.flatten4D = BooleanVar(value=False)   
        flatten4D_checkbox = Checkbutton(settings_frame, text='Sum 3Dt to 2Dt', variable=self.flatten4D)
        flatten4D_checkbox.grid(row=current_row, column=0)

        label = Label(settings_frame, text='Upsampleing factor:')
        label.grid(row=current_row, column=1, columnspan=2)
        self.upsampling_factor_spinbox = Spinbox(settings_frame, values=upsampling_values, width=4)
        self.upsampling_factor_spinbox.set(upsampling_values[2])
        self.upsampling_factor_spinbox.grid(row=current_row, column=3)
        current_row += 1

        self.snow_threshold_spinbox = Spinbox(settings_frame, from_=0, to=0.99, width=4, increment=0.1, format='%.2f')
        self.snow_threshold_spinbox.set(0.9)
        self.snow_threshold_spinbox.grid(row=current_row, column=3)

        self.remove_snow = BooleanVar(value=False)
        remove_snow_checkbox = Checkbutton(settings_frame, text='remove high value pixels above x*max_value, x =', variable=self.remove_snow)
        remove_snow_checkbox.grid(row=current_row, column=0, columnspan=3)
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

        ird_label = Label(ranges_frame, text='time ranges (no batch, default all):')
        ird_label.grid(row=current_row, column=0, columnspan=3)
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
        
        ird_label = Label(ird_frame, text='ird settings:')
        ird_label.grid(row=current_row, column=0, columnspan=3)
        current_row += 1

        self.ird_2d_averaging = IntVar(value=100)
        ird_2d_averaging_label = Label(ird_frame, text='2D averaging:')
        ird_2d_averaging_label.grid(row=current_row, column=0)
        ird_2d_averaging_spinbox = Spinbox(ird_frame, from_=1, to=100000, width=6, textvariable=self.ird_2d_averaging)
        ird_2d_averaging_spinbox.set(self.ird_2d_averaging.get())
        ird_2d_averaging_spinbox.grid(row=current_row, column=1)
        current_row += 1

        # data buttons
        open_button = Button(root, text='Open Image', command=self.open_image)
        open_button.grid(row=1, column=0, columnspan=1)
        
        process = Button(root, text='Process Image', command=self.process)
        process.grid(row=1, column=1)

    def X_correction_flipflop_fdml(self):
        if self.do_x_correction.get() == True:
            self.do_x_correction.set(False)
            self.do_FDML_correction.set(True)
    
    def X_correction_flipflop_galvo(self):
        if self.do_FDML_correction.get() == True:
            self.do_FDML_correction.set(False)
            self.do_x_correction.set(True)
                
    def decide_data_type(self):
        self.is_tiff = False
        self.is_ird = False
        self.is_single_frame = False
        self.is_single_volume = False
        self.is_2D_video = False
        self.is_3D_video = False
        try:
            del self.original_t_dim
        except AttributeError:
            pass

        if self.filename.endswith('.ird'):
            self.is_ird = True
            import rawdata
            import napari_streamin.arrays   
            self.ird_file = rawdata.InputFile()
            self.ird_file.open(self.filename)
            self.provider = rawdata.ImageDataProvider(self.ird_file,0)
            images = napari_streamin.arrays.VolumeArray(self.provider)

            if images.shape[1]==1:
                images = napari_streamin.arrays.ImageArray(self.provider)
                images.image_averaging = self.ird_2d_averaging.get()
                self.is_2D_video = True
            else:
                self.is_3D_video = True
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
        
        self.upsampling_factor = int(self.upsampling_factor_spinbox.get())
        self.melt = self.remove_snow.get()
        self.snow_threshold = float(self.snow_threshold_spinbox.get())

        for self.filename in self.filenames:
            if len(self.filenames)>1: # deactivate ranges for multiple files and reset ranges from previous file
                self.ranges = [] 

            self.decide_data_type()
            print("Processing: '"+self.filename+"' \nloading data")

            if self.is_single_frame or self.is_single_volume:
                with tiff.TiffFile(self.filename) as tif:
                    data = tif.asarray()
                    new_shape = self.create_new_array(data)[0]
                    if self.melt:
                        snow_value = np.amax(data)
                        data = self.melt_snow(data,snow_value)
                    if self.is_single_frame:
                        remapped_image = self.process_2D(data,new_shape)
                    elif self.is_single_volume:
                        remapped_image = self.process_3D(data,new_shape)
                    print('processing done')
                    self.save_image(remapped_image)

            elif self.is_3D_video:
                self.process_4D()
            elif self.is_2D_video:
                self.process_2Dt()
            else:
                print('Image dimension not supported!')
            if self.is_ird:
                self.ird_file.close()
            self.text_ranges.delete(2.0, END)

        
        
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
                pass
            try:
                t_dim = data.shape[-4]
            except IndexError:
                pass
        else:
            x_dim = data.shape[-1]
            y_dim = data.shape[-2]
            t_dim = data.shape[-3]
 
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
                    pass

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
        return new_array, memmap


    def load_ird(self, sections, snow_value=0, in_memmap=False, is2Dt=False):
        import napari_streamin.arrays
        match is2Dt:
            case False:
                irdata = napari_streamin.arrays.VolumeArray(self.provider)
                self.axes = 'QQYX'
            case True:
                irdata = napari_streamin.arrays.ImageArray(self.provider)
                irdata.image_averaging = self.ird_2d_averaging.get()
                self.axes = 'QYX'

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

        # melt snow if selected
        if self.melt:
            for timestep in range(t_dim):
                data[timestep] = self.melt_snow(data[timestep],snow_value)
                if timestep % 50 == 0:
                    print('removed snow in '+str(timestep)+' Volumes')

        print('Creating tif with corrected aspect ratio')
        new_shape,out_memmap = self.create_new_array(data)

        # process data
        print('correcting for sin distorsion')
        if self.do_z_correction.get() or self.do_y_correction.get() or self.do_x_correction.get() or self.do_FDML_correction.get():
            start=timer()
            for timestep in range(t_dim):
                new_shape[timestep] = self.process_3D(data[timestep],new_shape[0])
                if timestep % 50 == 0:
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
            # data = flatten_4D(data)
            # print(data.shape)
            new_shape = flatten_4D(new_shape)
            print(new_shape.shape)
        
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
                
        for timestep in np.arange(t_dim): 
            new_shape[timestep] = self.process_2D(data[timestep],new_shape[0])
            if timestep % 50 == 0:
                print('Frame '+str(timestep)+' corrected')

        self.save_data(data,new_shape,in_memmap,out_memmap)          
        return


    def process_3D(self,data,shape_array):
        x = self.do_x_correction.get()
        FDML = self.do_FDML_correction.get()
        y = self.do_y_correction.get()
        z = self.do_z_correction.get()

        if x or FDML:
            remapped_data = np.zeros((data.shape[0],data.shape[1],shape_array.shape[2]),dtype = 'uint16')
            remapped_data = np.swapaxes(remapped_data,1,2)
            shape_array = np.swapaxes(shape_array,1,2)
            data = np.swapaxes(data,1,2)

            if FDML:
                remapped_data = remapping3D(data,shape_array,self.upsampling_factor,FDML=True)
            else:
                remapped_data = remapping3D(data,shape_array,self.upsampling_factor)
            
            shape_array = np.swapaxes(shape_array,1,2)
            data = remapped_data
            data = np.swapaxes(data,1,2)

        if y:
            remapped_data = np.zeros((data.shape[0],shape_array.shape[1],data.shape[2]),dtype = 'uint16')
            remapped_data = remapping3D(data,shape_array,self.upsampling_factor)
            data = remapped_data

        if z:
            remapped_data = np.zeros((shape_array.shape[0],data.shape[1],data.shape[2]),dtype = 'uint16')
            shape_array = np.swapaxes(shape_array,0,1)
            remapped_data = np.swapaxes(remapped_data,0,1)
            data = np.swapaxes(data,0,1)
            
            remapped_data = remapping3D(data,shape_array,self.upsampling_factor)
            
            data=remapped_data
            data = np.swapaxes(data,0,1)
        return data

    def process_2D(self,data,shape_array):
        if not self.do_x_correction.get() and not self.do_y_correction.get():
            remapped_image = data

        if self.do_y_correction.get():
            remapped_image = remapping2D(data,shape_array,self.upsampling_factor)
            data=remapped_image
            
        if self.do_x_correction.get() or self.do_FDML_correction.get():
            shape_array = np.swapaxes(shape_array,0,1)
            remapped_image = np.swapaxes(data,0,1)
            if self.do_FDML_correction.get():
                remapped_image = remapping2D(remapped_image,shape_array,self.upsampling_factor,FDML=True)
            else:
                remapped_image = remapping2D(remapped_image,shape_array,self.upsampling_factor)
            remapped_image = np.swapaxes(remapped_image,0,1)
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
                else:
                    filtered_data[flakes] = 0

        return filtered_data
    
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
        return axes

    def save_image(self,data):
        print('compressing and saving data')

        outfile = self.filename.replace('.tif', '_processed.tif').replace('.ird', '_processed.tif')
        axes = self.get_axes()
        if self.flatten4D.get():
            axes = 'TYX'

        tiff.imwrite(
        outfile,
        data,
        ome=TRUE,
        compression=('zlib', 6),
        metadata={
        'axes': axes,
        })
        del data
        print('Data compressed and saved')

    def compress_image(self,path):
        print('attempting data compression')
        try:
            with tiff.TiffFile(path) as tif:
                data = tif.asarray()
                axes = self.get_axes()
                if self.flatten4D.get():
                    axes = 'TYX'
                tiff.imwrite(self.filename.replace('.tif','_processed.tif'),
                            data,
                            ome=TRUE,
                            compression=('zlib', 6),
                            resolution=(1/float(self.micron_y.get()), 1/float(self.micron_x.get())),
                            resolutionunit='MICROMETER',
                            metadata={
                            # 'spacing': 0.5,
                            # 'unit': 'um',
                            # 'finterval': 1 / 30,
                            # 'fps': 30.0,
                            'axes': axes,
                            })
            print('Data compressed and saved')
        except np.core._exceptions._ArrayMemoryError:
            print('Data too large for RAM, saved uncompressed data instead')
        return                        

if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()

# %%
