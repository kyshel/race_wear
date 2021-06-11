# funcs: convert result from yolo to submit
# notice: use for 14belt only

import argparse
import json
import os
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--json',  type=str, default='', help='json file, need cooco format')
opt = parser.parse_args()
fp_json = opt.json



# load coco_list
with open(fp_json) as f:
    yolo_list = json.load(f)

# load fn-id map
with open('link_image_fn_id.json') as f:
    map = json.load(f)

# xywh > xyxy
for i,val in enumerate(yolo_list):
    x,y,w,h = yolo_list[i]['bbox']
    x2,y2 = x+w,y+h
    yolo_list[i]['bbox'] = [x,y,x2,y2]

# # # only keep score that >= 0.2
# cutted_list = [val for i, val in enumerate(yolo_list) if val['score'] >= 0.2]
# yolo_list = cutted_list


# remove cls_id=0
# yolo_list = [ ele for ele in  yolo_list if ele['category_id'] != 0 ]


# image_id  fn2int
yolo_list_debug = copy.deepcopy(yolo_list)
for i,val in enumerate(yolo_list):
    fn_no_ext = yolo_list[i]['image_id']
    yolo_list[i]['image_id'] = map[fn_no_ext] # fn_no_ext > int_id
    yolo_list_debug[i]['image_id'] = (map[fn_no_ext], fn_no_ext)   # denug


fn_json = os.path.basename(fp_json)
fn_json_no_ext = os.path.splitext(fn_json)[0]
fn_out = fn_json_no_ext + '_submit.json'
fp_out =   fn_out



with open(fp_out, 'w') as fp:
    json.dump(yolo_list, fp )
    # json.dump(to_submit, fp, indent=4)


print('saved, check: ', fp_out)

# with open('submit_debug.json', 'w') as fp:
#     json.dump(yolo_list_debug, fp )
#     # json.dump(to_submit, fp, indent=4)