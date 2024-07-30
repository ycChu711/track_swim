import cv2
import torch
from deep_sort_pytorch.deep_sort import DeepSort
import deep_sort_pytorch.deep_sort.sort.track as track
import importlib

importlib.reload(track)

# Initialize DeepSORT
deepsort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")


def process_result(result):
    '''
    Description:
    Process the detection results to extract bounding box coordinates, confidence scores, and class indices
    Arguments:
        result: list, list of dictionaries containing the detection results
    Returns:
        bbox_xywh: torch.Tensor, tensor containing the bounding box coordinates
        confs: torch.Tensor, tensor containing the confidence scores
        class_idx: torch.Tensor, tensor containing the class indices
    '''
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
    ''' 
    Description:
    Calculate the intersection over union (IoU) of two bounding boxes
    Arguments:
        box_a: list, bounding box coordinates [x1, y1, x2, y2]
        box_b: list, bounding box coordinates [x1, y1, x2, y2]
    Returns:
        iou: float, intersection over union (IoU) value
    '''
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

def filter_overlapping_detections(detections, iou_threshold=0.3):
    '''
    Description:
    Filter overlapping detections based on the IoU threshold and class_id
    Arguments:
        detections: list, list of detections
        iou_threshold: float, IoU threshold
    Returns:
        filtered_detections: list, list of filtered detections
    '''
    filtered_detections = []
    skip_indices = set()
    for i, det1 in enumerate(detections):
        if i in skip_indices:
            continue

        keep = True
        for j, det2 in enumerate(detections):
            if i >= j or j in skip_indices:
                continue
            # Compare only detections with the same class_id
            if det1[4] == det2[4] and iou(det1[:4], det2[:4]) > iou_threshold:
                # Keep the detection with the larger area
                area1 = (det1[2] - det1[0]) * (det1[3] - det1[1])
                area2 = (det2[2] - det2[0]) * (det2[3] - det2[1])
                if area1 <= area2:
                    keep = False
                    skip_indices.add(i)
                    break
                else:
                    skip_indices.add(j)
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

    Arguments:
        bbox_left: int, x-coordinate of the top-left corner of the bounding box
        bbox_top: int, y-coordinate of the top-left corner of the bounding box
        bbox_right: int, x-coordinate of the bottom-right corner of the bounding box
        bbox_bottom: int, y-coordinate of the bottom-right corner of the bounding box
    
    Returns:
        center_x: int, x-coordinate of the center point
        center_y: int, y-coordinate of the center point
    '''
    center_x = (bbox_left + bbox_right) / 2
    center_y = (bbox_top + bbox_bottom) / 2

    return center_x, center_y

def generate_unused_id(id_to_lane_mapping, deepsort):
    '''
    Description:
    Generate a new ID that is not used by any existing track
    Arguments:
        id_to_lane_mapping: dict, mapping of track IDs to lane names
        deepsort: DeepSort, DeepSORT tracker object
    Returns:
        new_id: int, new track ID
    '''
    # Combine IDs from id_to_lane_mapping and DeepSORT's tracker
    used_ids = set(id_to_lane_mapping.keys()) | {track.track_id for track in deepsort.tracker.tracks}
    # Start with the highest existing ID + 1 or 1 if no IDs exist
    new_id = max(used_ids, default=0) + 1
    
    return new_id

def get_current_id_for_track(original_id, original_to_current_id_mapping):
    '''
    Description:
    Get the current ID for a track based on the original ID
    Arguments:
        original_id: int, original track ID
        original_to_current_id_mapping: dict, mapping of original track IDs to current track IDs
    Returns:
        current_id: int, current track ID
    '''
    return original_to_current_id_mapping.get(original_id, original_id)

def update_track_id_and_lane(identity, object_area_name, id_to_lane_mapping, original_to_current_id_mapping):
    '''
    Description:
    Update the track ID and lane name based on the object area
    
    Arguments:
        identity: int, original track ID
        object_area_name: str, name of the object area
        id_to_lane_mapping: dict, mapping of track IDs to lane names
        original_to_current_id_mapping: dict, mapping of original track IDs to current track IDs
    Returns:
        updated_identity: int, updated track ID
    '''
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

def draw_bounding_box(im, bbox_left, bbox_top, bbox_right, bbox_bottom, class_name, curr_identity, id_to_lane_mapping, frame_num):
    '''
    Description:
    Draw the bounding box and label on the image.

    Arguments:
        im: np.array, input image
        bbox_left: int, x-coordinate of the top-left corner of the bounding box
        bbox_top: int, y-coordinate of the top-left corner of the bounding box
        bbox_right: int, x-coordinate of the bottom-right corner of the bounding box
        bbox_bottom: int, y-coordinate of the bottom-right corner of the bounding box
        class_name: str, class name
        curr_identity: int, current track ID
        id_to_lane_mapping: dict, mapping of track IDs to lane names
    '''
    # Set box color based on class name, red for dangerous, green for swim
    if class_name == 'dangerous':
        box_color = (0, 0, 255)
    elif class_name == 'swim':
        box_color = (0, 255, 0)
    # Draw the bounding box in red
    cv2.rectangle(im, (int(bbox_left), int(bbox_top)), (int(bbox_right), int(bbox_bottom)), box_color, 2)
    # include original id and current id and lane i label
    label = f'{class_name}, \nID: {curr_identity}, \nLane: {id_to_lane_mapping[curr_identity]}, \nFrame: {frame_num}'
    #cv2.putText(im, label, (int(bbox_left), int(bbox_top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    lines = label.split('\n')
    start_x = int(bbox_left)
    start_y = int(bbox_top) - 10
    line_height = 20  # Adjust based on font size

    for i, line in enumerate(lines):
        line_y = start_y - (len(lines) - i) * line_height
        cv2.putText(im, line, (start_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)