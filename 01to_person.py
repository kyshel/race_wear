# make submit.json

# todo: add cut_items(crops_list,cls_id,threshold)
# match id
# src 0guard 1yes 2man 3no
# dst 0man 1guard 2yes 3no

from _func import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='best_predictions.json', help='json file, need cooco format')
    opt = parser.parse_args()
    fp_json = opt.json

    crops_list = read_json(fp_json)
    crops_list = cut_bigger(crops_list,threshold=0.2) # rm <= 0.2
    # crops_list = cut_items(crops_list,  2,  0.5 ) # 2man, 0.5threshold
    crops_list = add_man_tag(crops_list)
    stacked_dict = get_stacked_dict(crops_list)
    # stacked_dict = remove_overlapped(stacked_dict,overlapped_id=0) # iou 0guard
    # stacked_dict = remove_overlapped(stacked_dict, overlapped_id=1)  # iou 1yes
    # stacked_dict = remove_overlapped(stacked_dict, overlapped_id=3)  # iou 3no
    stacked_dict = wear2person(stacked_dict) # wear2person
    crops_list = rebuild_coco_from_stacked_dict(stacked_dict)
    crops_list = label_all_man(crops_list)
    crops_list = remove_man_tag(crops_list)
    make_json(fp_json,crops_list,stacked_dict)




exit()




















import json
import  argparse


def checkIntersection(boxA, boxB):
    # Check for boxA and boxB intersection, xywh
    # print(boxA,boxB)
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y

    foundIntersect = True
    if w < 0 or h < 0:
        foundIntersect = False

    return foundIntersect

    # return (foundIntersect, [x, y, w, h])


def check_overlap(crop_info,crops_list):
    found = 0
    overlaps_list = []
    is_overlap = False

    for crop_b in crops_list:
        if crop_b['category_id'] in [0,3] : # is_person: 0ground, 3fly
            # boxA = [crop_info["bbox"],crop_info["bbox"],crop_info["bbox"],crop_info["bbox"]]
            # boxB = [crop_b["bbox"],crop_b["bbox"],crop_b["bbox"],crop_b["bbox"]]
            if checkIntersection(crop_info["bbox"],crop_b["bbox"]):
                overlaps_list += [crop_b]
                found += 1

    if found != 0:
        # print('founded:' )
        # print(overlaps_list)
        is_overlap = True, overlaps_list
    else:
        print("warning! no overlaps found , image_id is " + str(crop_info["image_id"]) )

    return is_overlap, overlaps_list


def check_overlap_safebelt(crop_info,crops_list):
    found = 0
    overlaps_list = []
    is_overlap = False

    for crop_b in crops_list:
        if crop_b['category_id'] in [2] : # is_safebelt
            # boxA = [crop_info["bbox"],crop_info["bbox"],crop_info["bbox"],crop_info["bbox"]]
            # boxB = [crop_b["bbox"],crop_b["bbox"],crop_b["bbox"],crop_b["bbox"]]
            if checkIntersection(crop_info["bbox"],crop_b["bbox"]):
                overlaps_list += [crop_b]
                found += 1

    if found != 0:
        # print('founded:' )
        # print(overlaps_list)
        is_overlap = True
    else:
        print("warning! no overlaps found , image_id is " + str(crop_info["image_id"]) )

    return is_overlap, overlaps_list

def check_overlap_guard(crop_info,crops_list):
    found = 0
    overlaps_list = []
    is_overlap = False

    for crop_b in crops_list:
        if crop_b['category_id'] in [1] : # is_safebelt
            # boxA = [crop_info["bbox"],crop_info["bbox"],crop_info["bbox"],crop_info["bbox"]]
            # boxB = [crop_b["bbox"],crop_b["bbox"],crop_b["bbox"],crop_b["bbox"]]
            if checkIntersection(crop_info["bbox"],crop_b["bbox"]):
                overlaps_list += [crop_b]
                found += 1

    if found != 0:
        # print('founded:' )
        # print(overlaps_list)
        is_overlap = True
    else:
        print("warning! no overlaps found , image_id is " + str(crop_info["image_id"]) )

    return is_overlap, overlaps_list

def check_overlap_sky(crop_info,crops_list):
    found = 0
    overlaps_list = []
    is_overlap = False

    for crop_b in crops_list:
        if crop_b['category_id'] in [3] : # is_safebelt
            # boxA = [crop_info["bbox"],crop_info["bbox"],crop_info["bbox"],crop_info["bbox"]]
            # boxB = [crop_b["bbox"],crop_b["bbox"],crop_b["bbox"],crop_b["bbox"]]
            if checkIntersection(crop_info["bbox"],crop_b["bbox"]):
                overlaps_list += [crop_b]
                found += 1

    if found != 0:
        # print('founded:' )
        # print(overlaps_list)
        is_overlap = True
    else:
        print("warning! no overlaps found , image_id is " + str(crop_info["image_id"]) )

    return is_overlap, overlaps_list



def get_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1={'x1':box1[0],
         'y1':box1[1],
         'x2':box1[2]+box1[0],
         'y2':box1[3]+box1[1]}
    bb2={'x1':box2[0],
         'y1':box2[1],
         'x2':box2[2]+box2[0],
         'y2':box2[3]+box2[1]}


    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou,intersection_area



def read_json(fp_json):
    with open(fp_json) as f:
        json_list = json.load(f)
    return json_list

def cut_bigger(submit_list,threshold = 0.2):
    # # only keep score that >= 0.2
    cutted_list = [val for i, val in enumerate(submit_list) if val['score'] >= threshold]

    return  cutted_list

def get_stacked_dict(submit_list):
    # make stacked_dict
    stacked_dict = {}
    for i, val in enumerate(submit_list):
        if val['image_id'] in stacked_dict:
            stacked_dict[val['image_id']] += [val]
        else:
            stacked_dict[val['image_id']] = [val]
    # print(json.dumps(stacked_dict, indent=4))
    return stacked_dict


def remove_overlapped(stacked_dict,overlapped_id):
    # remove  overlap, max_score, seems like iou threshold
    print('[]removing belt overlapped')
    for key, crops_list in stacked_dict.items():
        # print(key)

        # get belt_max_score
        belt_max_score = 0
        for i, crop_info in enumerate(crops_list):
            # get belt_max_score
            if crop_info["category_id"] is overlapped_id:
                is_overlap, overlaps_list = check_overlap_by_id(crop_info, crops_list)
                if is_overlap:
                    # print("cls_id,score,box: ", crop_info["category_id"], crop_info['score'], crop_info['bbox'],
                    #     [(crop['category_id'], crop['score'], crop['bbox']) for crop in overlaps_list])

                    belt_max_score = crop_info['score']
                    for crop_overlapped in overlaps_list:
                        if crop_overlapped['score'] < belt_max_score:
                            stacked_dict[key].remove(crop_overlapped)
                        else:
                            belt_max_score = crop_overlapped['score']
                else:
                    print('no overlap to remove')  # this never exec as  you much overlap yourself!

        # print(belt_max_score)
        # print()
    return stacked_dict

def check_overlap_by_id(crop_info,crops_list):
    found = 0
    overlaps_list = []
    is_overlap = False
    crop_a_id = crop_info['category_id']

    for crop_b in crops_list:
        if crop_b['category_id'] is crop_a_id : # is_safebelt
            # boxA = [crop_info["bbox"],crop_info["bbox"],crop_info["bbox"],crop_info["bbox"]]
            # boxB = [crop_b["bbox"],crop_b["bbox"],crop_b["bbox"],crop_b["bbox"]]
            if checkIntersection(crop_info["bbox"],crop_b["bbox"]):
                overlaps_list += [crop_b]
                found += 1

    if found != 0:
        # print('founded:' )
        # print(overlaps_list)
        is_overlap = True
    else:
        print("warning! no overlaps found , image_id is " + str(crop_info["image_id"]) )

    return is_overlap, overlaps_list

def check_overlap_man(crop_info,crops_list):
    found = 0
    overlaps_list = []
    is_overlap = False

    for crop_b in crops_list:
        if crop_b['category_id'] in [2] : # is_person: 2
            # boxA = [crop_info["bbox"],crop_info["bbox"],crop_info["bbox"],crop_info["bbox"]]
            # boxB = [crop_b["bbox"],crop_b["bbox"],crop_b["bbox"],crop_b["bbox"]]
            if checkIntersection(crop_info["bbox"],crop_b["bbox"]):
                overlaps_list += [crop_b]
                found += 1

    if found != 0:
        # print('founded:' )
        # print(overlaps_list)
        is_overlap = True, overlaps_list
    else:
        print("warning! no overlaps found , image_id is " + str(crop_info["image_id"]) )

    return is_overlap, overlaps_list


def wear2person(stacked_dict):
    # to_person, max_area
    print('[]object to_person...')
    for key, crops_list in stacked_dict.items():
        # print(key)
        for i, crop_info in enumerate(crops_list):
            # print(crop_info)
            if crop_info["category_id"] in [0 , 1, 3]:  # no_person: 0guard 1yes_wear, 3no_wear
                is_overlap, overlaps_list = check_overlap_man(crop_info, crops_list)

                # select final person by IOU
                if is_overlap:
                    # print("cls_id,score,box: ", crop_info["category_id"], crop_info['score'], crop_info['bbox'],
                    #       [(crop['category_id'], crop['score'], crop['bbox']) for crop in overlaps_list])

                    crop_dict1 = stacked_dict[key][i]

                    bb1 = crop_dict1['bbox']

                    max_area = 0
                    max_iou = 0
                    max_score = 0
                    max_bbox = [1, 2, 3, 4]
                    for crop_dict2 in overlaps_list:
                        bb2 = crop_dict2['bbox']

                        iou, area = get_iou(bb1, bb2)
                        score = crop_dict2['score']
                        # print("iou_ratio, area: ",iou,area)

                        if area < max_area:  # there is optimize room here
                            pass
                        else:
                            max_area = area
                            max_bbox = bb2

                    stacked_dict[key][i]['bbox'] = max_bbox
                    # print("selected max_bbox",max_bbox)

                else:  # remove not overlapped
                    print('this object has no person overlapped, will remove this object!')
                    stacked_dict[key].remove(crop_info)
    return  stacked_dict

def rebuild_coco_from_stacked_dict(stacked_dict):
    # rebuild coco_format from stacked list
    final_list = []
    for key, crops_list in stacked_dict.items():
        for i, crop_info in enumerate(crops_list):
            final_list += [crop_info]
    return final_list



def make_json(fp_json,final_list):
    import os

    fn_json = os.path.basename(fp_json)
    fn_json_no_ext = os.path.splitext(fn_json)[0]
    fn_out = fn_json_no_ext + '_2_overlap.json'
    # make file that has orgin format
    with open(fn_out, 'w') as fp:
        json.dump(final_list, fp)
        # json.dump(to_submit, fp, indent=4)
    print('saved, check: ', fn_out)

    # remove cls_id=0
    final_list = [ele for ele in final_list if ele['category_id'] != 2] # remove 2man
    fn_out = fn_json_no_ext + '_3_remove_man.json'
    with open(fn_out, 'w') as fp:
        json.dump(final_list, fp)
        # json.dump(to_submit, fp, indent=4)
    print('saved, check: ', fn_out)

    ##########################################  step 2
    import copy

    # load coco_list
    yolo_list = final_list

    # load fn-id map
    with open('link_image_fn_id.json') as f:
        map_fn_id = json.load(f)

    # get not detected
    no_label_imgs = []
    for fn, id in map_fn_id.items():
        if fn not in stacked_dict:
            no_label_imgs += [fn]

    print('no labeld images list : ', no_label_imgs)

    # xywh > xyxy
    for i, val in enumerate(yolo_list):
        x, y, w, h = yolo_list[i]['bbox']
        x2, y2 = x + w, y + h
        yolo_list[i]['bbox'] = [x, y, x2, y2]

    # # # only keep score that >= 0.2
    # cutted_list = [val for i, val in enumerate(yolo_list) if val['score'] >= 0.2]
    # yolo_list = cutted_list

    # image_id  fn2int
    yolo_list_debug = copy.deepcopy(yolo_list)
    for i, val in enumerate(yolo_list):
        fn_no_ext = yolo_list[i]['image_id']
        yolo_list[i]['image_id'] = map_fn_id[fn_no_ext]  # fn_no_ext > int_id
        yolo_list_debug[i]['image_id'] = (map_fn_id[fn_no_ext], fn_no_ext)  # denug

    # steps
    #  guard > man, yes > man, no > man

    # match id
    # src 0guard 1yes 2man 3no
    # dst 0man 1guard 2yes 3no
    for i, ele in enumerate(yolo_list):
        if ele['category_id'] == 0:
            yolo_list[i]['category_id'] = 1
        elif ele['category_id'] == 1:
            yolo_list[i]['category_id'] = 2
        elif ele['category_id'] == 2:
            yolo_list[i]['category_id'] = 0

    fp_out = fn_json_no_ext + '_4_submit.json'

    with open(fp_out, 'w') as fp:
        json.dump(yolo_list, fp)
        # json.dump(to_submit, fp, indent=4)

    print('saved, check: ', fp_out)


    # match id
    # src 0guard 1yes 2man 3no
    # dst 0man 1guard 2yes 3no
# need : rm man < 0.5 (have miss)
####################3 main ############################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='best_predictions.json', help='json file, need cooco format')
    opt = parser.parse_args()
    fp_json = opt.json

    crops_list = read_json(fp_json)
    crops_list = cut_bigger(crops_list,threshold=0.2) # rm <= 0.2
    stacked_dict = get_stacked_dict(crops_list)
    # stacked_dict = remove_overlapped(stacked_dict,overlapped_id=0) # iou 0guard
    # stacked_dict = remove_overlapped(stacked_dict, overlapped_id=1)  # iou 1yes
    # stacked_dict = remove_overlapped(stacked_dict, overlapped_id=3)  # iou 3no
    stacked_dict = wear2person(stacked_dict) # wear2person
    crops_list = rebuild_coco_from_stacked_dict(stacked_dict)
    make_json(fp_json,crops_list)









exit()


parser = argparse.ArgumentParser()
parser.add_argument('--json',  type=str, default='best_predictions.json', help='json file, need cooco format')
opt = parser.parse_args()
fp_json = opt.json

with open(fp_json) as f:
    submit_list = json.load(f)

# # only keep score that >= 0.2
cutted_list = [val for i, val in enumerate(submit_list) if val['score'] >= 0.2]
submit_list = cutted_list



# make stacked_dict
stacked_dict = {}
for i,val in enumerate(submit_list):
    if val['image_id'] in stacked_dict:
        stacked_dict[val['image_id']] += [val]
    else:
        stacked_dict[val['image_id']] = [val]
# print(json.dumps(stacked_dict, indent=4))


# remove belt overlap, max_score, seems like iou threshold
print('[]removing belt overlapped')
for key, crops_list in stacked_dict.items():
    # print(key)

    # get belt_max_score
    belt_max_score = 0
    for i, crop_info in enumerate(crops_list):
        # get belt_max_score
        if crop_info["category_id"] in [2]:
            is_overlap, overlaps_list = check_overlap_safebelt(crop_info, crops_list)
            if is_overlap:
                # print("cls_id,score,box: ", crop_info["category_id"], crop_info['score'], crop_info['bbox'],
                #     [(crop['category_id'], crop['score'], crop['bbox']) for crop in overlaps_list])

                belt_max_score = crop_info['score']
                for crop_overlapped in overlaps_list:
                    if crop_overlapped['score'] < belt_max_score:
                        stacked_dict[key].remove(crop_overlapped)
                    else:
                        belt_max_score =  crop_overlapped['score']
            else:
                print('no safebelt overlap') # this never exec as  you much overlap yourself!

    # print(belt_max_score)
    # print()


# remove guard overlap, max_score, seems like iou threshold
print('[]removing guard overlapped')
for key, crops_list in stacked_dict.items():
    # print(key)

    # get belt_max_score
    belt_max_score = 0
    for i, crop_info in enumerate(crops_list):
        # get belt_max_score
        if crop_info["category_id"] in []:
            is_overlap, overlaps_list = check_overlap_guard(crop_info, crops_list)
            if is_overlap:
                # print("cls_id,score,box: ", crop_info["category_id"], crop_info['score'], crop_info['bbox'],
                #     [(crop['category_id'], crop['score'], crop['bbox']) for crop in overlaps_list])

                belt_max_score = crop_info['score']
                for crop_overlapped in overlaps_list:
                    if crop_overlapped['score'] < belt_max_score:
                        stacked_dict[key].remove(crop_overlapped)
                    else:
                        belt_max_score =  crop_overlapped['score']
            else:
                print('no safebelt overlap') # this never exec as  you much overlap yourself!





# remove sky overlap, max_score, seems like iou threshold
print('[]removing sky overlapped')
for key, crops_list in stacked_dict.items():
    # print(key)

    # get belt_max_score
    belt_max_score = 0
    for i, crop_info in enumerate(crops_list):
        # get belt_max_score
        if crop_info["category_id"] in [3]:
            is_overlap, overlaps_list = check_overlap_sky(crop_info, crops_list)
            if is_overlap:
                # print("cls_id,score,box: ", crop_info["category_id"], crop_info['score'], crop_info['bbox'],
                #     [(crop['category_id'], crop['score'], crop['bbox']) for crop in overlaps_list])

                belt_max_score = crop_info['score']
                for crop_overlapped in overlaps_list:
                    if crop_overlapped['score'] < belt_max_score:
                        stacked_dict[key].remove(crop_overlapped)
                    else:
                        belt_max_score =  crop_overlapped['score']
            else:
                print('no safebelt overlap') # this never exec as  you much overlap yourself!

    # print(belt_max_score)
    # print()



# to_person, max_area
print('[]object to_person...')
for key, crops_list in stacked_dict.items():
    # print(key)
    for i, crop_info in enumerate(crops_list):
        # print(crop_info)
        if crop_info["category_id"] in [1, 2]:  # no_preson: 1guard, 2safebelt
            is_overlap,overlaps_list = check_overlap(crop_info,crops_list)

            # select final person by IOU
            if is_overlap:
                # print("cls_id,score,box: ", crop_info["category_id"], crop_info['score'], crop_info['bbox'],
                #       [(crop['category_id'], crop['score'], crop['bbox']) for crop in overlaps_list])

                crop_dict1 = stacked_dict[key][i]

                bb1 = crop_dict1['bbox']

                max_area = 0
                max_iou =  0
                max_score = 0
                max_bbox = [1,2,3,4]
                for crop_dict2 in overlaps_list:
                    bb2 = crop_dict2['bbox']

                    iou, area =get_iou(bb1, bb2)
                    score =  crop_dict2['score']
                    # print("iou_ratio, area: ",iou,area)


                    if area < max_area:   # there is optimize room here
                        pass
                    else:
                        max_area = area
                        max_bbox = bb2


                stacked_dict[key][i]['bbox'] = max_bbox
                # print("selected max_bbox",max_bbox)

            else:  #  remove not overlapped
                print('this object has no person overlapped, will remove this object!')
                stacked_dict[key].remove(crop_info)




# rebuild coco_format from stacked list
final_list = []
for key, crops_list in stacked_dict.items():
    for i, crop_info in enumerate(crops_list):
        final_list += [crop_info]


import os


fn_json = os.path.basename(fp_json)
fn_json_no_ext = os.path.splitext(fn_json)[0]
fn_out = fn_json_no_ext + '_2_overlap.json'
# make file that has orgin format
with open(fn_out, 'w') as fp:
    json.dump(final_list, fp )
    # json.dump(to_submit, fp, indent=4)
print('saved, check: ', fn_out)



# remove cls_id=0
final_list = [ ele for ele in  final_list if ele['category_id'] != 0 ]
fn_out = fn_json_no_ext + '_3_remove_ground.json'
with open(fn_out, 'w') as fp:
    json.dump(final_list, fp )
    # json.dump(to_submit, fp, indent=4)
print('saved, check: ', fn_out)




##########################################  step 2
import copy

# load coco_list
yolo_list = final_list

# load fn-id map
with open('link_image_fn_id.json') as f:
    map_fn_id = json.load(f)



# get nothing detected
no_label_imgs = []
for fn,id in map_fn_id.items():
    if fn not in stacked_dict:
        no_label_imgs += [fn]

print('no labeld images list : ',no_label_imgs)


# xywh > xyxy
for i,val in enumerate(yolo_list):
    x,y,w,h = yolo_list[i]['bbox']
    x2,y2 = x+w,y+h
    yolo_list[i]['bbox'] = [x,y,x2,y2]

# # # only keep score that >= 0.2
# cutted_list = [val for i, val in enumerate(yolo_list) if val['score'] >= 0.2]
# yolo_list = cutted_list




# image_id  fn2int
yolo_list_debug = copy.deepcopy(yolo_list)
for i,val in enumerate(yolo_list):
    fn_no_ext = yolo_list[i]['image_id']
    yolo_list[i]['image_id'] = map_fn_id[fn_no_ext] # fn_no_ext > int_id
    yolo_list_debug[i]['image_id'] = (map_fn_id[fn_no_ext], fn_no_ext)   # denug

# steps
#  guard > man, yes > man, no > man


# match id
# src 0guard 1yes 2man 3no
# dst 0man 1guard 2yes 3no
for i,ele in enumerate(yolo_list):
    if ele['category_id'] == 0:
        yolo_list[i]['category_id'] = 1
    elif ele['category_id'] == 1:
        yolo_list[i]['category_id'] = 2
    elif ele['category_id'] == 2:
        yolo_list[i]['category_id'] = 0



fp_out = fn_json_no_ext + '_4_submit.json'




with open(fp_out, 'w') as fp:
    json.dump(yolo_list, fp )
    # json.dump(to_submit, fp, indent=4)


print('saved, check: ', fp_out)

