#%%
from tkinter import filedialog

import itk
import tifffile as tiff
import numpy as np
#read images
#%%
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
    rigid_params = parameters.GetDefaultParameterMap("bspline") # can be 'translation', 'rigid', 'nonrigid', 'affine', 'bspline'
    # affine_params = parameters.GetDefaultParameterMap("affine")

    # for key in bspline_params:
    #     print(key, bspline_params[key])
    #Modify some
    # rigid_params["Registration"] = ["MultiResolutionRegistration"]
    rigid_params["ResultImageFormat"] = ["tiff"]
    # rigid_params["resolution"] = ["1", "0.255", "0.45"]
    # affine_params["resolution"] = ["1", "0.255", "0.45"]
    rigid_params["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    # affine_params["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    parameters.AddParameterMap(rigid_params)
    # parameters.AddParameterMap(affine_params)
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

    for i in range(np.ceil((np_stack.shape[0]/2)).astype(int)):
        volume = itk.GetImageFromArray(np_stack[i*2,:,:,:])
        transformed_slice = itk.transformix_filter(volume, result_parameters)
        np_stack[i*2, :, :, :] = itk.GetArrayFromImage(transformed_slice)
    return np_stack

#%%
result_image, result_parameters = register(fixed_image_path, moving_image_path)
#%%
transformed_stack = transform_stack(stack_path, result_parameters)
transformed_stack[transformed_stack < 4] = 0


tiff.imwrite(stack_path.replace(".ome.tif", "_transformed.ome.tif"),
            transformed_stack,
            photometric='minisblack',
            metadata={'axes': 'TZYX'},
            ome=True,
            compressionargs=('zlib', 6)
            )

# %%
