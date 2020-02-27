
import sys
import cv2
import torch
from kitti import Kitti
from lyft import Lyft
from homo_adap_dataset import HomoAdapDataset
import viz

def sfm_inspector(data):
    img = torch.cat((data["refs"][0], data["tgt"], data["refs"][1]), dim=1)
    cv2.imshow("img", viz.tensor2img(img))
    cv2.imshow("gt_sparse", viz.tensor2depthimg(data["gt_sparse"]))

def simple_inspector(data):
    img = data["img"]
    warp = data["warp"]
    homography = data["homography"]
    print(homography)
    cv2.imshow("img", viz.tensor2img(img))
    cv2.imshow("warp", viz.tensor2img(warp))

def main():

    choise = sys.argv[1]
    path = sys.argv[2]

    print(choise, path)

    if choise == "kitti":
        dataset = Kitti(path)
        inspector = sfm_inspector
    elif choise == "lyft":
        dataset = Lyft(path)
        inspector = sfm_inspector
    elif choise == "homo":
        dataset = HomoAdapDataset(path)
        inspector = simple_inspector
    else:
        print("No such choise: %s" % choise)
        exit()

    print(len(dataset))

    for i, data in enumerate(dataset, start=0):

        print(i)

        inspector(data)
        
        key = cv2.waitKey(0)
        if key == 27:
            break

if __name__ == '__main__':
    main()