
import mayavi.mlab as mlab
import numpy as np


def draw_rgb_points(fig, xyz, rgb):
  pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], np.arange(xyz.shape[0]),
    mode="point", figure=fig)
  pts.glyph.color_mode = 'color_by_scalar'
  alpha = np.ones((rgb.shape[0], 1))
  rgba = np.concatenate((rgb, alpha), axis=1)
  pts.module_manager.scalar_lut_manager.lut.table = (rgba * 255).astype(np.uint8)

def draw_points(fig, xyz, values, cmap="gnuplot"):
  mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], values,
    mode="point", colormap=cmap, figure=fig)

def draw_rgb_spheres(fig, xyz, rgb):
  src = mlab.pipeline.scalar_scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
  alpha = np.ones((rgb.shape[0], 1))
  rgba = np.concatenate((rgb, alpha), axis=1)
  rgba = (rgba * 255).astype(np.uint8)
  src.add_attribute(rgba, 'colors')
  src.data.point_data.set_active_scalars('colors')
  g = mlab.pipeline.glyph(src)
  g.glyph.scale_mode = 'data_scaling_off'
  g.glyph.glyph.scale_factor = 0.01

if __name__ == "__main__":


  fig = mlab.figure(figure=None, bgcolor=(0,0,0),
    fgcolor=None, engine=None, size=(1000, 500))

  N = 10000
  pc = np.random.rand(N, 3)

  colors = np.random.uniform(size=(N, 3))

  draw_rgb_points(fig, pc, colors)
  #draw_rgb_spheres(fig, pc, colors)
  #draw_points(fig, pc, pc[:,2])

  mlab.show()