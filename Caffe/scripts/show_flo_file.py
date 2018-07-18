import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import cv2, time, glob
from common import readFlow, readTupleList, computeColor


def get_example_fig():
    height = 151
    width  = 151
    truerange = 1
    range_s = truerange * 1.04
    s2 = round(float(height)/2)

    x = np.asarray(range(1, width+1), dtype=np.float32)
    y = np.asarray(range(1, height+1), dtype=np.float32)
    x = np.tile(x, (height, 1))
    y = np.tile(y, (width, 1))
    y = np.transpose(y, (1,0))
    u = x*range_s/s2 - range_s
    v = y*range_s/s2 - range_s

    img = computeColor(u/truerange, v/truerange)
    img[s2, :, :] = 0
    img[:, s2, :] = 0

    # cv2.namedWindow('img')
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('img')
    # cv2.imwrite('../data/optical_flow_colImg.png', img)
    plt.figure(1)
    plt.axis('off')
    plt.imshow(img)
    # plt.show()
    # plt.close(1)


def show_trituple(trituple_list, save_flow, save_path):
    trituple = readTupleList(trituple_list)
    for name1, name2, flo_name in trituple:
        img1 = cv2.imread(name1)
        img2 = cv2.imread(name2)
        img1 = img1[:, :, [2, 1, 0]]
        img2 = img2[:, :, [2, 1, 0]]
        plt.figure(2)
        plt.subplot(2,2,1), plt.axis('off')
        plt.title('img0'),  plt.imshow(img1)
        plt.subplot(2,2,2), plt.axis('off')
        plt.title('img1'),  plt.imshow(img2)

        F = readFlow(flo_name)
        u = F[:, :, 0]
        v = F[:, :, 1]
        u_range = max(u.max(), abs(u.min()))
        v_range = max(v.max(), abs(v.min()))
        truerange = max(u_range, v_range, 1)
        img_flow = computeColor(u/truerange, v/truerange)
        plt.subplot(2, 2, (3, 4))
        plt.axis('off')
        plt.title('optical flow')
        plt.imshow(img_flow)
        file_name = osp.splitext(osp.basename(flo_name))[0]

        if save_flow:
            save_name = osp.join(save_path, file_name+'.png')
            cv2.imwrite(save_name, img_flow[:,:,[2,1,0]])
            time.sleep(0.01)
        # plt.show()
        # plt.savefig(osp.join(save_path, file_name + '_fig.png'))


def show_flow_in_dir(flow_path, save_flow=False):
    flo_files = glob.glob(osp.join(flow_path, '*.flo'))
    flo_files.sort()
    for i, flo_name in enumerate(flo_files, start=1):
        print('processing: {}/{}'.format(i, len(flo_files)))
        F = readFlow(flo_name)
        u = F[:, :, 0]
        v = F[:, :, 1]
        u_range = max(u.max(), abs(u.min()))
        v_range = max(v.max(), abs(v.min()))
        truerange = max(u_range, v_range, 1)
        # truerange = 20
        img_flow = computeColor(u / truerange, v / truerange)
        plt.figure(2)
        plt.axis('off')
        plt.title('optical flow')
        plt.imshow(img_flow)

        if save_flow:
            save_name = osp.splitext(flo_name)[0] + '.png'
            cv2.imwrite(save_name, img_flow[:,:,[2,1,0]])
            time.sleep(0.01)


def show_tvl1_trituple(trituple_list, save_flow, save_path):
    trituple = readTupleList(trituple_list)
    for name1, name2, flo_name in trituple:
        img1 = cv2.imread(name1)
        img2 = cv2.imread(name2)
        img1 = img1[:, :, [2, 1, 0]]
        img2 = img2[:, :, [2, 1, 0]]
        plt.figure(2)
        plt.subplot(2,2,1), plt.axis('off')
        plt.title('img0'),  plt.imshow(img1)
        plt.subplot(2,2,2), plt.axis('off')
        plt.title('img1'),  plt.imshow(img2)

        width  = img1.shape[1]
        height = img1.shape[0]
        # f = open(name1+'_x.flo', 'rb')
        # u1 = np.fromfile(f, np.float32, width * height).reshape((height, width))
        # f = open(name1+'_y.flo', 'rb')
        # v1 = np.fromfile(f, np.float32, width * height).reshape((height, width))
        f = open(name1+'.flo', 'rb')
        F = np.fromfile(f, np.float32, width * height * 2).reshape((2, height, width))
        u = F[0, :, :]
        v = F[1, :, :]
        u_range = max(u.max(), abs(u.min()))
        v_range = max(v.max(), abs(v.min()))
        truerange = max(u_range, v_range, 1)
        img_flow = computeColor(u/truerange, v/truerange)
        plt.subplot(2, 2, (3, 4))
        plt.axis('off')
        plt.title('optical flow')
        plt.imshow(img_flow)
        plt.show()
        file_name = osp.splitext(osp.basename(name1))[0]

        if save_flow:
            save_name = osp.join(save_path, file_name+'_tvl1.png')
            cv2.imwrite(save_name, img_flow[:,:,[2,1,0]])
            time.sleep(0.01)


if __name__ == '__main__':
    # get_example_fig()
    save_flow = True

    ## show flow file in trituple.list
    # trituple_list = '../data/surveillance_examples/trituple.list'
    # save_path = '../data/surveillance_examples/res_s_my'
    # show_trituple(trituple_list, save_flow, save_path)

    ## show flow file in dir
    flow_path = "./"
    show_flow_in_dir(flow_path, save_flow)

    ## show tvl1-flow file in trituple.list
    # trituple_list = '../data/surveillance_examples/trituple.list'
    # save_path = '../data/surveillance_examples/res_tvl1'
    # show_tvl1_trituple(trituple_list, save_flow, save_path)

