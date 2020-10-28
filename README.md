
pytorch-ignite       0.2.1     
torch                1.3.1     
torchvision          0.4.2

pytorch-ignite       0.2.1     
torch                1.6.0     
torchvision          0.7.0
# sfmnet

python main.py homo-train --log-interval 10 --lr 0.0001 --name homo_a1_m --dataset kitti_homo_adapt --batch 128 --epochs 300 --load-point checkpoints/point_kitti_a1_m/epoch_30

python main.py homo-train --log-interval 10 --lr 0.0001 --name homo_a1_m_450_1 --dataset kitti_homo_adapt --batch 32 --epochs 300 --load-point checkpoints/point_kitti_a1_m/epoch_30.pt

python main.py homo-train --log-interval 10 --lr 0.0005 --name homo_a1_m_450_2 --dataset kitti_homo_adapt --batch 32 --epochs 300 --load-point checkpoints/point_kitti_a1_m/epoch_30.pt

python main.py homo-train --log-interval 10 --lr 0.00001 --name homo_a1_m_450_4 --dataset kitti_homo_adapt --batch 32 --epochs 300 --load-point checkpoints/point_kitti_a1_m/epoch_30.pt

## Train
python main.py point-train --log-interval 100 --lr 0.001 --name pointuni2
python main.py point-train --log-interval 100 --lr 0.0001 --dataset cocokittylyft_homo_adapt --name point_all2

python main.py point-train --log-interval 100 --lr 0.0001 --dataset kitti_homo_adapt --name point_kitti_a1
python main.py point-train --log-interval 100 --lr 0.0001 --dataset kitti_homo_adapt --name point_kitti_a1_m --epochs 100 --load checkpoints/point_kitti_a1/epoch_30.pt
python main.py point-train --log-interval 100 --lr 0.005 --dataset kitti_homo_adapt --name point_kitti_a2
python main.py point-train --log-interval 100 --lr 0.00005 --dataset kitti_homo_adapt --name point_kitti_a3
python main.py point-train --log-interval 100 --lr 0.0002 --dataset kitti_homo_adapt --name point_kitti_a4 --epochs 100

### devide by W:
python main.py homo-train --log-interval 10 --lr 0.00001 --name homo_a1_m_450_7 --dataset kitti_homo_adapt --batch 32 --epochs 300 --load-point checkpoints/point_kitti_a1_m/epoch_30.pt

### devide by W and reg:
python main.py homo-train --log-interval 10 --lr 0.00001 --name homo_a1_m_450_9 --dataset kitti_homo_adapt --batch 32 --epochs 300 --load-point checkpoints/point_kitti_a1_m/epoch_30.pt

### point train that works pretty good
python main.py point-train --log-interval 100 --lr 0.00003 --dataset kitti_homo_adapt --epochs 300 --name point_kitti_m6

### point train with more uniform weight
python main.py point-train --log-interval 100 --lr 0.00005 --dataset kitti_homo_adapt --epochs 300 --name point_kitti_m9


python main.py fcons-train --log-interval 500 --lr 0.0005 --name fc29 --load-point checkpoints/pointuni2/epoch_30.pt
python main.py fcons-train --log-interval 500 --lr 0.0001 --name fc30 --load-point checkpoints/pointuni2/epoch_30.pt

## Debug
python main.py point-debug --load checkpoints/pointuni2/epoch_30.pt
python main.py fcons-debug --load checkpoints/fc29/epoch_1.pt 

### Nice results
python main.py homo-synth-debug --load-consensus checkpoints/homo_synth_49/epoch_136.pt --dataset synth_homo_points

## Homo that works
python main.py homo-synth-train --log-interval 10 --lr 0.0001 --name homo_synth_36 --dataset synth_homo_points --batch 128 --epochs 300
python main.py homo-synth-train --log-interval 10 --lr 0.0001 --name homo_synth_46 --dataset synth_homo_points --batch 128 --epochs 300

## Test point
python main.py point-test --load checkpoints/point_kitti_m10/epoch_30.pt --dataset kitti_homo_adapt
unsup_RS -> mean: 0.796, std: 0.051
unsup_LE -> mean: 0.667, std: 0.081
orb_RS -> mean: 0.844, std: 0.073
orb_LE -> mean: 0.762, std: 0.072

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