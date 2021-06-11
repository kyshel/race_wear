# save labeled image (for 14belt ony, as has overlapped)

# usage
# python 03slm.py  --json  best_predictions2.json  --test  test0/
# python 03slm.py  --json  best_predictions.json  --test  test/
# python 03slm.py  --json  best_predictions2.json  --test  test/  --save save_test2/


import  argparse
import os
from os import listdir
from os.path import isfile, join
import cv2
import json
from tqdm import tqdm

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3,class_id=''):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), color, thickness=tl, lineType=cv2.LINE_AA) # c1 changed!
    x[0] = x[2] if class_id in [1,2] else x[0]  # for 14belt only
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],thickness=tf, lineType=cv2.LINE_AA)


def get_stacked_dict(coco_list):
    # make stacked_dict
    stacked_dict = {}
    for i, val in enumerate(coco_list):
        if val['image_id'] in stacked_dict:
            stacked_dict[val['image_id']] += [val]
        else:
            stacked_dict[val['image_id']] = [val]
    # print(json.dumps(stacked_dict, indent=4))
    return stacked_dict


def read_dir(dir1):
    # make file full path list from dir_name
    fp_list = []
    for fn in listdir(dir1):
        fp = dir1 + fn

        if isfile(fp):
            fp_list += [fp]
    return fp_list


def xywh2xyxy(box):
    xyxy = [box[0],
            box[1],
            box[2] + box[0],
            box[3] + box[1]]
    return xyxy


def check():
    # check save_dir if exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def label_image():
    with open(coco_fp) as f:
        coco_list = json.load(f)
    stacked_dict = get_stacked_dict(coco_list)

    im_fp_list = read_dir(test_dir)

    # way1 read 1 write 1
    pbar = tqdm(total=len(im_fp_list), position=0, leave=True,
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    no_label_list = []
    for im_fp in im_fp_list:
        im = cv2.imread(im_fp)
        im0 = im
        fn = os.path.basename(im_fp)
        fn_no_ext = os.path.splitext(fn)[0]

        if fn_no_ext in stacked_dict:
            for crop in stacked_dict[fn_no_ext]:
                xyxy = xywh2xyxy(crop['bbox'])
                c = crop['category_id']
                # label = names[c]
                conf = crop['score']
                label = f'{names[c]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                             line_thickness=opt.line_thickness, class_id = c)

            cv2.imwrite(save_dir + fn, im0)
        else:
            no_label_list += [im_fp]
        pbar.update(1)
    pbar.close()

    # save no labeld images filename to txt
    if len(no_label_list)!=0:
        fn_json = os.path.basename(coco_fp)
        fn_json_no_ext = os.path.splitext(fn_json)[0]
        fn_txt = fn_json_no_ext + '_no_labeled_imgs.txt'
        fp_txt = os.path.dirname(coco_fp) + '/' + fn_txt
        with open(fp_txt, "w") as output:
            output.write(str(no_label_list))
        print('Warning! no_label_images  cnt: ', len(no_label_list),', details saved to: ',fp_txt)
    print('Labeled images have saved to:', save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',  type=str, default='', help='json file, need cooco format')
    parser.add_argument('--test', type=str, default='', help='input test dir')
    parser.add_argument('--save', type=str, default='', help='output save dir')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    print(opt)

    coco_fp = opt.json
    test_dir = opt.test
    fn_json_no_ext = os.path.splitext(os.path.basename(coco_fp))[0]
    save_dir= ( fn_json_no_ext + '/' if opt.save == '' else opt.save)
    names = {0: 'ground', 1: 'guard', 2: 'safebelt', 3: 'sky'}

    check()
    label_image()

    # usage
    # python 03slm.py  --json  best_predictions2.json  --test  test0/
    # python 03slm.py  --json  best_predictions.json  --test  test/
    # python 03slm.py  --json  best_predictions2.json  --test  test/  --save save_test2/





