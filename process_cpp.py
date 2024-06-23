#%%
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import subprocess
import os

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Processing')
        self.root.resizable(True, True)

        self.label = Label(root, text='Upsampling factor X:')
        self.label.grid(row=0, column=1)
        self.upsampling_factor_X_spinbox = Spinbox(root, from_=1, to=100, width=4)
        self.upsampling_factor_X_spinbox.set(3)
        self.upsampling_factor_X_spinbox.grid(row=0, column=2)

        self.label = Label(root, text='Upsampling factor Y:')
        self.label.grid(row=1, column=1)
        self.upsampling_factor_Y_spinbox = Spinbox(root, from_=1, to=100, width=4)
        self.upsampling_factor_Y_spinbox.set(3)
        self.upsampling_factor_Y_spinbox.grid(row=1, column=2)

        self.label = Label(root, text='Upsampling factor Z:')
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
        self.is2D_video_checkbox.grid(row=3, column=0)

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

        self.open_button = Button(root, text='Open Image', command=self.open_image)
        self.open_button.grid(row=5, column=0, columnspan=1)
        self.process_button = Button(root, text='Process Image', command=self.process_image)
        self.process_button.grid(row=5, column=2)

        self.remapped_image = None

    def open_image(self):
        self.filename = filedialog.askopenfilename()
        print(f"Selected file: {self.filename}")

    def process_image(self):
        upsampling_factor_X = int(self.upsampling_factor_X_spinbox.get())
        upsampling_factor_Y = int(self.upsampling_factor_Y_spinbox.get())
        upsampling_factor_Z = int(self.upsampling_factor_Z_spinbox.get())
        is2D = self.is2D_video.get()
        melt = self.remove_snow.get()
        snow_threshold = float(self.snow_threshold_spinbox.get())
        do_x_correction = self.do_x_correction.get()
        do_y_correction = self.do_Y_correction.get()
        do_z_correction = self.do_z_correction.get()
        try_gpu = self.try_GPU.get()

        if self.filename:
            # Construct the command to call the C++ executable
            command = [
                "./image_processing", 
                self.filename,
                str(upsampling_factor_X),
                str(upsampling_factor_Y),
                str(upsampling_factor_Z),
                str(int(is2D)),
                str(int(melt)),
                str(snow_threshold),
                str(int(do_x_correction)),
                str(int(do_y_correction)),
                str(int(do_z_correction)),
                str(int(try_gpu))
            ]

            print("Running command: " + " ".join(command))

            # Call the C++ executable
            subprocess.run(command)
        else:
            print("No file selected!")

if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()

# %%
