# sfmnet

## Train ckmmands


## Training data junk

du -a | grep velodyne_points/data | cut -d/ -f3 | sort | uniq -c | sort -nr | grep " 1 "

du -a | grep image_03/data | cut -d/ -f3 | sort | uniq -c | sort -nr | grep " 1 "

du -a | grep image_02/data | cut -d/ -f3 | sort | uniq -c | sort -nr | grep " 1 "


0000000177.bin

ls -l /home/ai/Code/Data/kitti_raw/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/ | cut -d " " -f11 | sort | less


/home/ai/Code/Data/kitti_raw/2011_10_03/2011_10_03_drive_0058_sync/velodyne_points/data/0000000028.bin


ai@ai-System-Product-Name:~/Code/sfmnet/datasets/kitti$
ai@ai-System-Product-Name:~/Code/sfmnet/datasets/kitti$ python prepare_train_data.py ~/Code/Data/kitti_raw/ --dataset-format kitti --static-frames static_frames.txt --with-depth --with-pose --dump-root ~/Code/Data/kitti2 --num-threads 8
Found 151 potential scenes
Retrieving frames
  7%|███████▎                                                                                                      | 10/151 [17:37<12:12:58, 311.90s/it]^[[A^[[B/home/ai/Code/sfmnet/datasets/kitti/kitti_raw_loader.py:160: UserWarning: genfromtxt: Empty input file: "/home/ai/Code/Data/kitti_raw/2011_09_26/2011_09_26_drive_0059_sync/oxts/data/0000000000.txt"
  metadata = np.genfromtxt(f)
 14%|███████████████▍                                                                                               | 21/151 [29:54<6:22:06, 176.35s/it]pebble.common.RemoteTraceback: Traceback (most recent call last):
  File "/home/ai/Enviroments/torch/lib/python3.6/site-packages/pebble/common.py", line 174, in process_execute
    return function(*args, **kwargs)
  File "prepare_train_data.py", line 46, in dump_example
    for sample in data_loader.get_scene_imgs(scene_data):
  File "/home/ai/Code/sfmnet/datasets/kitti/kitti_raw_loader.py", line 208, in get_scene_imgs
    yield construct_sample(scene_data, i, frame_id)
  File "/home/ai/Code/sfmnet/datasets/kitti/kitti_raw_loader.py", line 190, in construct_sample
    sample['depth'] = self.generate_depth_map(scene_data, i)
  File "/home/ai/Code/sfmnet/datasets/kitti/kitti_raw_loader.py", line 281, in generate_depth_map
    velo = np.fromfile(velo_file_name, dtype=np.float32).reshape(-1, 4)
FileNotFoundError: [Errno 2] No such file or directory: Path('/home/ai/Code/Data/kitti_raw/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/0000000177.bin')


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "prepare_train_data.py", line 124, in <module>
    main()
  File "prepare_train_data.py", line 97, in main
    for _ in tqdm(tasks.result(), total=n_scenes):
  File "/home/ai/Enviroments/torch/lib/python3.6/site-packages/tqdm/std.py", line 1104, in __iter__
    for obj in iterable:
  File "/home/ai/Enviroments/torch/lib/python3.6/site-packages/pebble/pool/base_pool.py", line 211, in next
    raise result
FileNotFoundError: [Errno 2] No such file or directory: Path('/home/ai/Code/Data/kitti_raw/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/0000000177.bin')
 14%|███████████████                                                                                             | 21/151 [2:39:03<16:24:38, 454.45s/it]
ai@ai-System-Product-Name:~/Code/sfmnet/datasets/kitti$ 