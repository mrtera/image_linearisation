#%%
import numpy as np
import napari
import tifffile as tiff


def paint_cross(image_shape, position, size):

    cross = np.zeros(image_shape, dtype=int)
    z, y, x = position
    x_size = int(size*1/0.45)
    y_size = int(size*1/0.225)
    x_min = max(0, x - x_size)
    xmin=x
    x_max = min(position[2], x + x_size + 1)
    y_min = max(0, y - y_size)
    ymin=y
    y_max = min(position[1], y + y_size + 1)
    z_min = max(0, z - size)
    zmin=z
    z_max = min(position[0], z + size + 1)

    cross[:,y_min:y_max,x_min:x_max] = 1
    cross[z_min:z_max,:,x_min:x_max] = 1
    cross[z_min:z_max,y_min:y_max,:] = 1
    return cross

# with tiff.TiffFile('E:/streamin_20250205_173054_processed_best part prop-1-95 substack_pc_hmf.ome.tif') as tif:
#     image = tif.asarray()

cross = np.zeros_like(image)
image_shape = image[0].shape

for time in range(cross.shape[0]):
    pos=(time,*map(int, np.unravel_index(int(np.argmax(image[time])), image[time].shape)))[1:4]
    cross[time] = paint_cross(image_shape,pos, size=1)

cross[81] = paint_cross(image_shape,(41,316,267), size=1)
cross[82]=paint_cross(image_shape,(40,296,283), size=1)
cross[83]=paint_cross(image_shape,(40,250,304), size=1)

    
# viewer = napari.Viewer()
# viewer.add_image(image, name='image')
# viewer.add_image(cross, name='cross')
# napari.gui_qt()
with tiff.TiffWriter('E:/cross_painting_output0.tif', bigtiff=True) as tif:
    tif.save(cross,compression= 'zlib' )
