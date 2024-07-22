import cv2
import torch
from deep_sort_pytorch.deep_sort import DeepSort
import deep_sort_pytorch.deep_sort.sort.track as track
import importlib

importlib.reload(track)

# Initialize DeepSORT
deepsort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")


def process_result(result):
    bbox_xywh = []
    confs = []
    class_idx = []
    if result:
        for det in result[0]:
            bbox = det['bbox']
            score = det['score']
            class_id = det['class_idx']

            x1, y1, x2, y2 = map(int, bbox)

            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                bbox_xywh.append([x1, y1, w, h])
                confs.append(score)
                class_idx.append(class_id)

    bbox_xywh = torch.Tensor(bbox_xywh)
    confs = torch.Tensor(confs)
    class_idx = torch.Tensor(class_idx)

    return bbox_xywh, confs, class_idx

def iou(box_a, box_b):
    # Determine the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # Compute the area of both the prediction and ground-truth rectangles
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

def filter_overlapping_detections(detections, iou_threshold=0.5):
    filtered_detections = []
    for i, det1 in enumerate(detections):
        keep = True
        for j, det2 in enumerate(detections):
            if i >= j:
                continue
            if iou(det1[:4], det2[:4]) > iou_threshold:
                # Keep the detection with the higher confidence score
                keep = det1[4] > det2[4]
                if not keep:
                    break
        if keep:
            filtered_detections.append(det1)
    return filtered_detections

def fix_offset(bbox_left, bbox_top, bbox_right, bbox_bottom):
    '''
    Description:
    Fix the offset of the bounding box: the ouput of the model treat the top-left corner of the image as center
    of the image, so we need to fix the offset to get the correct bounding box coordinates.

    bbox_left: int, x-coordinate of the top-left corner of the bounding box
    bbox_top: int, y-coordinate of the top-left corner of the bounding box
    bbox_right: int, x-coordinate of the bottom-right corner of the bounding box
    bbox_bottom: int, y-coordinate of the bottom-right corner of the bounding box
    '''
    x_offset = int((bbox_right - bbox_left) / 2)
    y_offset = int((bbox_bottom - bbox_top) / 2)
    bbox_left = int(bbox_left + x_offset)
    bbox_top = int(bbox_top + y_offset)
    bbox_right = int(bbox_right + x_offset)
    bbox_bottom = int(bbox_bottom + y_offset)

    return bbox_left, bbox_top, bbox_right, bbox_bottom

def get_center_point(bbox_left, bbox_top, bbox_right, bbox_bottom):
    '''
    Description:
    Get the center point of the bounding box.
    '''
    center_x = (bbox_left + bbox_right) / 2
    center_y = (bbox_top + bbox_bottom) / 2

    return center_x, center_y

def generate_unused_id(id_to_lane_mapping, deepsort):
    # Combine IDs from id_to_lane_mapping and DeepSORT's tracker
    used_ids = set(id_to_lane_mapping.keys()) | {track.track_id for track in deepsort.tracker.tracks}
    # Start with the highest existing ID + 1 or 1 if no IDs exist
    new_id = max(used_ids, default=0) + 1
    
    return new_id

def get_current_id_for_track(original_id, original_to_current_id_mapping):
    return original_to_current_id_mapping.get(original_id, original_id)

def update_track_id_and_lane(identity, object_area_name, id_to_lane_mapping, original_to_current_id_mapping):
    current_id = get_current_id_for_track(identity, original_to_current_id_mapping)
    if current_id not in id_to_lane_mapping:
        id_to_lane_mapping[current_id] = object_area_name
    elif object_area_name != id_to_lane_mapping[current_id]:
        # Assign a new ID and update mappings
        new_id = generate_unused_id(id_to_lane_mapping, deepsort)
        original_to_current_id_mapping[identity] = new_id
        id_to_lane_mapping[new_id] = object_area_name
        return new_id
    return current_id

def draw_bounding_box(im, bbox_left, bbox_top, bbox_right, bbox_bottom, class_name, curr_identity, id_to_lane_mapping):
    '''
    Description:
    Draw the bounding box and label on the image.
    '''
    cv2.rectangle(im, (int(bbox_left), int(bbox_top)), (int(bbox_right), int(bbox_bottom)), (0, 0, 255), 2)
    # include original id and current id and lane i label
    label = f'{class_name}, \nID: {curr_identity}, \nLane: {id_to_lane_mapping[curr_identity]}'
    #cv2.putText(im, label, (int(bbox_left), int(bbox_top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    lines = label.split('\n')
    start_x = int(bbox_left)
    start_y = int(bbox_top) - 10
    line_height = 20  # Adjust based on font size

    for i, line in enumerate(lines):
        line_y = start_y - (len(lines) - i) * line_height
        cv2.putText(im, line, (start_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)