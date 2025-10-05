#%%
from tkinter import filedialog

import itk
import tifffile as tiff
import numpy as np
#read images

fixed_image_path = filedialog.askopenfilename(title="Select fixed image")
moving_image_path = filedialog.askopenfilename(title="Select moving image")
stack_path = filedialog.askopenfilename(title="Select stack to be transformed")
#%%
# register two images as master
def register(fixed_image_path, moving_image_path):
    fixed_image = itk.imread(fixed_image_path)
    moving_image = itk.imread(moving_image_path)
    #set up parameters
    parameters = itk.ParameterObject.New()
    translation_params = parameters.GetDefaultParameterMap("rigid")
    translation_params["ResultImageFormat"] = ["tiff"]
    translation_params["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    translation_params["MaximumNumberOfIterations"] = ["2000", "2000", "2000"]  # Set to a reasonable number of iterations 2000 for good
    parameters.AddParameterMap(translation_params)
    print(" image shape: ", fixed_image.shape)
    #Perform Registration
    registered_image, result_parameters = itk.elastix_registration_method(
        fixed_image, moving_image, parameter_object=parameters)
    print(result_parameters.GetParameterMap(0)["TransformParameters"])
    return registered_image, result_parameters


# apply same shift with transformix
def transform_stack(stack_path, result_parameters):
    with tiff.TiffFile(stack_path) as tif:
        np_stack = tif.asarray()
        print("stack shape: ", np_stack.shape)
    
    for i in range(int(np_stack.shape[0]/2)):
        volume = itk.GetImageFromArray(np_stack[i*2,:,:,:])
        transformed_slice = itk.transformix_filter(volume, result_parameters)
        np_stack[i*2, :, :, :] = itk.GetArrayFromImage(transformed_slice)
    return np_stack

#%%
result_image, result_parameters = register(fixed_image_path, moving_image_path)
#%%
transformed_stack = transform_stack(stack_path, result_parameters)
#%%

tiff.imwrite("transformed_stack.ome.tif",
            transformed_stack,
            photometric='minisblack',
            metadata={'axes': 'TZYX'},
            ome=True,
            compressionargs=('zlib', 6)
            )

# %%
