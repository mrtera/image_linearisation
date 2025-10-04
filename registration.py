#%%
from tkinter import filedialog

import itk
import tifffile as tiff
#%%
#read images


def register(fixed_image_path, moving_image_path):
    fixed_image = itk.imread(fixed_image_path)
    moving_image = itk.imread(moving_image_path)
    #set up parameters
    parameters = itk.ParameterObject.New()
    translation_params = parameters.GetDefaultParameterMap("translation")
    translation_params["AutomaticTransformInitialization"] = ["false"]
    translation_params["InitialTransformParametersFileName"] = ["NoInitialTransform"]
    translation_params['AutomaticParameterEstimation'] = ['false']
    translation_params["ResultImageFormat"] = ["tiff"]
    translation_params["NumberOfResolutions"] = ["3"]
    translation_params["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    translation_params["MaximumNumberOfIterations"] = ["2000", "20000", "20000"]  # Set to a reasonable number of iterations 2000 for good
    parameters.AddParameterMap(translation_params)
    print(" image shape: ", fixed_image.shape)
    #Perform Registration
    registered_image, result_parameters = itk.elastix_registration_method(
        fixed_image, moving_image, parameter_object=parameters)
    print(result_parameters.GetParameterMap(0)["TransformParameters"])
    return registered_image, result_parameters

def transform_stack(stack_path, result_parameters):
    image = itk.imread(stack_path)
    np_stack = itk.GetArrayFromImage(image)
    print("stack shape: ", np_stack.shape)

    # return itk.transformix_filter(image, result_parameters)

# apply same shift with transformix
fixed_image_path = filedialog.askopenfilename(title="Select fixed image")
moving_image_path = filedialog.askopenfilename(title="Select moving image")
stack_path = filedialog.askopenfilename(title="Select stack to be transformed")

result_image, result_parameters = register(fixed_image_path, moving_image_path)
transformed_stack = transform_stack(stack_path, result_parameters)


tiff.imwrite("registered_transformix.ome.tif",
            transformed_stack,
            photometric='minisblack',
            metadata={'axes': 'ZYX'},
            ome=True,
            compressionargs=('zlib', 6)
            )
