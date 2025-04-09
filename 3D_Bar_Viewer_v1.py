#@ ImagePlus imp
# 3D_Bar_Viewer.py

from ij import IJ
import sys
import os
import subprocess

roi = imp.getRoi()
if roi is None:
    IJ.error("No ROI", "Please select a rectangular ROI first, then re-run.")
    sys.exit(0)

bounds = roi.getBounds()
x0 = bounds.x
y0 = bounds.y
width = bounds.width
height = bounds.height

ip = imp.getProcessor()
values_2d = []
for row in range(y0, y0 + height):
    row_values = []
    for col in range(x0, x0 + width):
        row_values.append(ip.getPixelValue(col, row))
    values_2d.append(row_values)

py_script = r"""import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

vals = np.array({vals})
h, w = vals.shape
x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_f = x_coords.ravel()
y_f = y_coords.ravel()
z_f = np.zeros_like(x_f)
dx = 0.8 * np.ones_like(x_f)
dy = 0.8 * np.ones_like(y_f)
dz = vals.ravel()

vals_norm = (dz - dz.min()) / (dz.max() - dz.min() + 1e-9)
colors = cm.viridis(vals_norm)

ax.bar3d(x_f, y_f, z_f, dx, dy, dz, shade=True, color=colors, edgecolor='none')
ax.view_init(elev=30, azim=-60)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Pixel Value')

plt.title("3D Bar Plot of ROI Pixel Values (Color-Coded)")
plt.show()
""".format(vals=values_2d)

temp_dir = IJ.getDirectory("temp")
if temp_dir is None:
    temp_dir = os.path.expanduser("~")

py_filepath = os.path.join(temp_dir, "fiji_3d_bar_plot.py")
with open(py_filepath, 'wb') as f:
    f.write(py_script.encode('utf-8'))

try:
    subprocess.call(["python", py_filepath])
except Exception as e:
    IJ.error("Error launching external Python", str(e))
