
import random
import sys
import cv2
import torch
from kitti import Kitti
from lyft import Lyft
from synthia import Synthia
from homo_adap_dataset import HomoAdapDataset, HomoAdapDatasetFromSequences, HomoAdapDatasetCocoKittiLyft
import viz
import numpy as np
import utils

click_pt = None
K = None

def mouse_callback(event, x, y, flags, param):
    global click_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        click_pt = np.array([x, y, 1])
        p = np.linalg.inv(K) @ click_pt
        print("K@click_p:", p)
        
def sfm_inspector(data):
    global K
    K = utils.torch_to_numpy(data["K"])
    img = torch.cat((data["refs"][0], data["tgt"], data["refs"][1]), dim=1)
    cv2.imshow("img", viz.tensor2img(img))
    cv2.imshow("gt_sparse", viz.tensor2depthimg(data["gt_sparse"]))
    #cv2.namedWindow("click")
    #cv2.setMouseCallback("click", mouse_callback)
    #cv2.imshow("click", viz.tensor2img(data["tgt"]))

def simple_inspector(data):
    img = data["img"]
    warp = data["warp"]
    homography = data["homography"]
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
    elif choise == "synthia":
        dataset = Synthia(path)
        inspector = sfm_inspector
    elif choise == "cocoa_homo_adapt":
        dataset = HomoAdapDataset(path)
        inspector = simple_inspector
    elif choise == "sequence_homo_adapt":
        dataset = HomoAdapDatasetFromSequences(path)
        inspector = simple_inspector
    elif choise == "cocokittylyft_homo_adapt":
        dataset = HomoAdapDatasetCocoKittiLyft(path)
        inspector = simple_inspector
    else:
        print("No such choise: %s" % choise)
        exit()

    print(len(dataset))

    #for i, data in enumerate(dataset, start=0):

    l = list(range(len(dataset)))
    ids = random.sample(l, len(l))

    for i in ids:

        print(i)

        data = dataset[i]

        loop = True
        while loop:
            key = cv2.waitKey(10)
            if key == 27:
                exit()
            elif key != -1:
                loop = False

            inspector(data)

if __name__ == '__main__':
    main()