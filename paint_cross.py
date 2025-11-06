#%%
import numpy as np
import napari
import tifffile as tiff
import csv

def read_cvsv_as_array(file_path):
    """
    Read a CSV where column A (col 0) = x, B (col 1) = y, C (col 2) = z.
    Values start at row 3 (A3/B3/C3). Returns an (N,3) numpy array of
    integer tuples in (z, y, x) order suitable for paint_cross.
    """
    positions = []
    with open(file_path, newline='') as fh:
        reader = csv.reader(fh)
        # skip first two rows (A1/B1/C1 and A2/B2/C2)
        for _ in range(2):
            next(reader, None)
        for row in reader:
            if not row:
                continue
            # ensure there are at least 3 columns
            if len(row) < 3:
                continue
            x_s, y_s, z_s = row[0].strip(), row[1].strip(), row[2].strip()
            if x_s == "" and y_s == "" and z_s == "":
                continue
            # try to parse numbers (accept floats, commas as decimal separators)
            try:
                xi = int(round(float(x_s.replace(',', '.'))))
                yi = int(round(float(y_s.replace(',', '.'))))
                zi = int(round(float(z_s.replace(',', '.'))))
            except Exception:
                # skip rows that don't parse
                continue
            positions.append((zi, yi, xi))
    return np.asarray(positions, dtype=int)

def paint_cross(image_shape, position, size):
    cross = np.zeros(image_shape, dtype=int)
    z, y, x = position
    x_size = int(size/0.467)
    y_size = int(size/0.26)
    z_size = int(size/1.59)

    x_min = max(0, x - x_size)
    # xmin=x
    x_max = min(position[2], x + x_size + 1)
    y_min = max(0, y - y_size)
    # ymin=y
    y_max = min(position[1], y + y_size + 1)
    z_min = max(0, z - size)
    # zmin=z
    z_max = min(position[0], z + size + 1)

    cross[:,y_min:y_max,x_min:x_max] = 1
    cross[z_min:z_max,:,x_min:x_max] = 1
    cross[z_min:z_max,y_min:y_max,:] = 1
    return cross

# with tiff.TiffFile('D:/streamin_20250205_173054_processed_best part prop-1-95 substack_pc_hmf.ome.tif') as tif:
#     imagein = tif.asarray()
# image = imagein[2:-4,:,:,:]  # Adjust indexing as needed

positions = read_cvsv_as_array('D:\\positions.csv')
cross = np.zeros_like(image)
image_shape = image[0].shape

for time in range(cross.shape[0]):
    pos = positions[time]
    cross[time] = paint_cross(image_shape,pos, size=1)
    
with tiff.TiffWriter('D:/cross_painting_output-avg-all.tif', bigtiff=True) as tif:
    tif.write(cross,
              compression= 'zlib',
              compressionargs={'level': 6})


# with tiff.TiffWriter('D:/cropped.tif', bigtiff=True) as tif:
#     tif.write(image,
#               compression= 'zlib',
#               compressionargs={'level': 6})
