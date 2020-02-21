
import sys
import cv2
import torch
from kitti import Kitti
from lyft import Lyft
import viz

def main():

    if sys.argv[1] == "kitti":
        dataset = Kitti(sys.argv[2])
    elif sys.argv[1] == "lyft":
        dataset = Lyft(sys.argv[2])

    for i in range(0, len(dataset)):

        data = dataset[i]

        #print(relative_transform(tgt_pose, ref_pose[0]))
        #print(relative_transform(tgt_pose, ref_pose[1]))
        #print("---")+

        img = torch.cat((data["refs"][0], data["tgt"], data["refs"][1]), dim=1)

        
        cv2.imshow("img", viz.tensor2img(img))
        cv2.imshow("gt_sparse", viz.tensor2depthimg(data["gt_sparse"]))
        
        key = cv2.waitKey(0)
        if key == 27:
            break

if __name__ == '__main__':
    main()