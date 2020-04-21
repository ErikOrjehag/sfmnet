
import os
import sys
import glob

def main():

  path = sys.argv[1]

  images02 = [im.replace("image_02", "xxx") for im in sorted(glob.glob(os.path.join(path, "*", "*", "image_02", "data", "*.png")))]
  images03 = [im.replace("image_03", "xxx") for im in sorted(glob.glob(os.path.join(path, "*", "*", "image_03", "data", "*.png")))]

  #diff = set(images02).symmetric_difference(images03)
  images = set(images02).union(set(images03))

  missing = []
  empty = []

  for im in images:
    velodyne = im.replace("xxx", "velodyne_points")[:-4] + ".bin"
    oxts = im.replace("xxx", "oxts")[:-4] + ".txt"
    if not os.path.isfile(velodyne) or not os.path.isfile(oxts):
      missing.append(im)
    if os.path.isfile(oxts):
      if os.stat(oxts).st_size == 0:
        empty.append(oxts)

  missing = sorted(missing)
  empty = sorted(empty)

  print("MISSING:")
  print("\n".join(missing))

  print("EMPTY:")
  print("\n".join(empty))

  """
  Missing velodyne/oxts for a few frames but fairly stationary:
  2011_09_26_drive_0009_sync/xxx/data/0000000177.png
  2011_09_26_drive_0009_sync/xxx/data/0000000178.png
  2011_09_26_drive_0009_sync/xxx/data/0000000179.png
  2011_09_26_drive_0009_sync/xxx/data/0000000180.png

  Missing velodyne/oxts for all frames:
  2011_09_28_drive_0225_sync
  2011_09_29_drive_0108_sync
  2011_09_30_drive_0072_sync
  2011_10_03_drive_0058_sync

  Oxts files are empty:
  2011_09_26_drive_0059_sync

  Missing images in 02 and 03:
  2011_09_26_drive_0119_sync

  Very few images and they are junk:
  du -a | grep jpg | cut -d/ -f2 | sort | uniq -c | sort -nr
  2011_09_28_drive_0039_sync
  2011_09_26_drive_0017_sync
  2011_09_28_drive_0034_sync
  2011_09_28_drive_0043_sync
  

  du -a | grep velodyne_points/data | cut -d/ -f3 | sort | uniq -c | sort -nr | grep " 1 "
  du -a | grep image_03/data | cut -d/ -f3 | sort | uniq -c | sort -nr | grep " 1 "
  du -a | grep image_02/data | cut -d/ -f3 | sort | uniq -c | sort -nr | grep " 1 "

  ADD THIS:

  ====> Test scenes ====>
  2011_09_28_drive_0225
  2011_09_29_drive_0108
  2011_09_30_drive_0072
  2011_10_03_drive_0058
  2011_09_26_drive_0119
  2011_09_26_drive_0059

  ====> Static frames ====>
  2011_09_26 2011_09_26_drive_0009_sync 0000000177
  2011_09_26 2011_09_26_drive_0009_sync 0000000178
  2011_09_26 2011_09_26_drive_0009_sync 0000000179
  2011_09_26 2011_09_26_drive_0009_sync 0000000180
  """

if __name__ == "__main__":
  main()