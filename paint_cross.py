#%%
import numpy as np
import napari
import tifffile as tiff
import csv

def center_mass(image):
    total = np.sum(image)
    if total == 0:
        return (0,0,0)
    z_indices, y_indices, x_indices = np.indices(image.shape)
    z_cm = int(np.round(np.sum(z_indices * image) / total))
    y_cm = int(np.round(np.sum(y_indices * image) / total))
    x_cm = int(np.round(np.sum(x_indices * image) / total))
    return (z_cm, y_cm, x_cm)

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
        print(positions)
    return np.asarray(positions, dtype=int)

def paint_cross(image_shape, position, size):
    cross = np.zeros(image_shape, dtype=int)
    z, y, x = position
    x_size = int(size/0.5)
    y_size = int(size/0.2)
    z_size = int(size/1)

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

def generate_trace(positions):
    trace = np.zeros((positions.shape[0],positions.shape[1]+2), dtype=int)
    for time in range(positions.shape[0]):
        z,y,x = positions[time]
        trace[time] = (0,time,z,y,x)
    return trace

with tiff.TiffFile('D:/streamin_20250205_173305_phase-corrected_hmf.ome.tif') as tif:
    image = tif.asarray()

cross = np.zeros_like(image)
image_shape = image[0].shape
# cm = np.zeros_like(image)

positions = read_cvsv_as_array('D:\\SMP positions-pixel.csv')
trace = generate_trace(positions)
for row in trace:
    print(f'[{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}],')

for time in range(cross.shape[0]):
    pos = positions[time]
    cross[time] = paint_cross(image_shape,pos, size=1)


# with tiff.TiffWriter('D:/MP_cross.tif', bigtiff=True) as tif:
#     tif.write(cross,
#               compression= 'zlib',
#               compressionargs={'level': 6})

# with tiff.TiffWriter('D:/cropped.tif', bigtiff=True) as tif:
#     tif.write(image,
#               compression= 'zlib',
#               compressionargs={'level': 6})

# with tiff.TiffWriter('D:/cross_center_mass.tif', bigtiff=True) as tif:
#     tif.write(cm,
#               compression= 'zlib',
#               compressionargs={'level': 6})