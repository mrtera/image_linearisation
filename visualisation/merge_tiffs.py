#%%
import os
import tifffile as tiff
import numpy as np
import glob
from tkinter import filedialog
def merge_tiffs_in_folder(input_folder, output_file, z_spacing=0.5):
    # Get list of all tiff files in the folder
    tiff_files = sorted(glob.glob(os.path.join(input_folder, '*.tif')))
    metadata = {}
    with tiff.TiffFile(tiff_files[0]) as tif:
        for key, value in tif.pages[0].tags.items():
            metadata[str(key)] = str(value.value)
        
    if not tiff_files:
        print("No TIFF files found in the specified folder.")
        return
    
    # Read all tiff files and stack them in z direction
    image_stack = []
    for file in tiff_files:
        with tiff.TiffFile(file) as tif:
            image = tif.asarray()
            image_stack.append(image)

    
    # Convert list to a 3D numpy array
    merged_stack = np.stack(image_stack, axis=0)  # shape: (z, y, x)
    
    # Save the merged stack as a new tiff file with metadata
    metadata['PhysicalSizeZ'] = str(z_spacing)
    metadata['PhysicalSizeZUnit'] = 'µm'
    tiff.imwrite(
        output_file, 
        merged_stack,
        metadata=metadata,
        ome=True,
        photometric='minisblack',
        compression='zlib',
        compressionargs={'level': 6}
        )

    
if __name__ == '__main__':
    input_folder = filedialog.askdirectory()
    output_file = 'a_merged_output.tif'
    z_spacing = 0.5
    merge_tiffs_in_folder(input_folder, f'{input_folder}\\{output_file}', z_spacing)