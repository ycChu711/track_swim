import cv2
import torch
import numpy as np
# Add lane_identification import
import lane_identification.lane_identification as li

# Add deep_sort import
from deep_sort_pytorch.deep_sort import DeepSort

import deep_sort_pytorch.deep_sort.sort.track as track
import importlib

# Reload the track module to ensure changes are recognized
importlib.reload(track)


import os
from tqdm import tqdm

# Initialize DeepSORT
deepsort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

def load_names(path):
    class_names = []
    with open(path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def letterbox_image(src, out_size):
    in_h, in_w = src.shape[:2]
    out_h, out_w = out_size
    scale = min(out_w / in_w, out_h / in_h)
    mid_h, mid_w = int(in_h * scale), int(in_w * scale)
    dst = cv2.resize(src, (mid_w, mid_h))
    top, down = (out_h - mid_h) // 2, (out_h - mid_h + 1) // 2
    left, right = (out_w - mid_w) // 2, (out_w - mid_w + 1) // 2
    dst = cv2.copyMakeBorder(dst, top, down, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return dst, [left, top, scale]

def scale_coordinates(data, pad_w, pad_h, scale, img_shape):
    for i in data:
        i['bbox'][0] = max(0, min(img_shape[1], (i['bbox'][0] - pad_w) / scale))
        i['bbox'][1] = max(0, min(img_shape[0], (i['bbox'][1] - pad_h) / scale))
        i['bbox'][2] = max(0, min(img_shape[1], (i['bbox'][2] - pad_w) / scale))
        i['bbox'][3] = max(0, min(img_shape[0], (i['bbox'][3] - pad_h) / scale))

def tensor2detection(offset_boxes, det):
    offset_box_vec, score_vec = [], []
    for i in range(offset_boxes.size(0)):
        offset_box_vec.append([int(offset_boxes[i, j]) for j in range(4)])
        score_vec.append(float(det[i, 4]))
    return offset_box_vec, score_vec

def post_processing(detections, pad_w, pad_h, scale, img_shape, conf_thres, iou_thres):
    conf_mask = detections[..., 4] >= conf_thres
    detections = detections[conf_mask]
    if not detections.size(0):
        return []
    detections[..., 5:] *= detections[..., 4:5]
    boxes = xywh2xyxy(detections[..., :4])
    max_conf_scores, max_conf_indices = torch.max(detections[..., 5:], dim=-1)
    detections = torch.cat((boxes, max_conf_scores.unsqueeze(1), max_conf_indices.unsqueeze(1)), dim=1)
    offset_box = detections[:, :4] + detections[:, 5:6] * 4096
    offset_box_vec, score_vec = tensor2detection(offset_box, detections)

    # Debug: Print the sizes of offset_box_vec and score_vec
    # print(f"offset_box_vec size: {len(offset_box_vec)}")
    # print(f"score_vec size: {len(score_vec)}")


    indices = cv2.dnn.NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres)

    # Debug: Print the content of indices
    # print(f"indices: {indices}")

    result = []
    if len(indices) > 0:  # Ensure there are valid indices
        for idx in indices.flatten():
            detection = {
                'bbox': offset_box_vec[idx],
                'score': score_vec[idx],
                'class_idx': int(detections[idx, 5])  # The class index is at position 5
            }
            result.append(detection)
    scale_coordinates(result, pad_w, pad_h, scale, img_shape)
    return [result]

def demo(img, detections, class_names):
    if detections:
        for det in detections[0]:
            bbox = det['bbox']
            score = det['score']
            class_idx = det['class_idx']

            # Ensure the bounding box coordinates are integers
            x1, y1, x2, y2 = map(int, bbox)

            # Debug: Print the bounding box coordinates
            # print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_names[class_idx]} {score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# load lane coordinates
def load_lane_coordinates(path):
    lane_coordinates = {}
    with open(path, 'r') as f:
        for line in f:
            name, coordinates = line.strip().split(':')
            coordinates = eval(coordinates)
            lane_coordinates[name] = coordinates
    return lane_coordinates

def process_function(im, module, class_names, device, deepsort, areas, id_area):
    conf_thres, iou_thres = 0.4, 0.5
    img_input, pad_info = letterbox_image(im, (640, 640))
    pad_w, pad_h, scale = pad_info
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(img_input).unsqueeze(0).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        detections = module(tensor_img)[0]

    result = post_processing(detections.to(device), pad_w, pad_h, scale, im.shape[:2], conf_thres, iou_thres)

    bbox_xywh, confs, class_idx = process_result(result)

    if bbox_xywh.numel() > 0:
        outputs, _ = deepsort.update(bbox_xywh, confs, class_idx, im)

        if len(outputs) > 0:
            for j, output in enumerate(outputs):
                bbox_left, bbox_top, bbox_right, bbox_bottom, class_id, identity = output

                bbox_left, bbox_top, bbox_right, bbox_bottom = fix_offset(bbox_left, bbox_top, bbox_right, bbox_bottom)

                center_x, center_y = get_center_point(bbox_left, bbox_top, bbox_right, bbox_bottom)

                object_area_name = li.assign_objects_to_areas(center_x, center_y, areas)

                identity = int(identity)

                if identity not in id_area.keys():
                    id_area[identity] = object_area_name
                elif identity in id_area.keys() and object_area_name != id_area[identity]:
                    remove_track(deepsort, identity)
                    continue

                draw_bounding_box(im, bbox_left, bbox_top, bbox_right, bbox_bottom, class_names[class_id], identity)

    return im

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

def remove_track(deepsort, identity):
    '''
    Description:
    Remove the track from the tracker and the dictionary of track IDs and areas.
    '''
    for track in deepsort.tracker.tracks:
        if track.track_id == identity:
            track.mark_missed()
            deepsort.tracker.tracks.remove(track)
            break

def draw_bounding_box(im, bbox_left, bbox_top, bbox_right, bbox_bottom, class_name, identity):
    '''
    Description:
    Draw the bounding box and label on the image.
    '''
    cv2.rectangle(im, (int(bbox_left), int(bbox_top)), (int(bbox_right), int(bbox_bottom)), (0, 0, 255), 2)
    label = f'{class_name}, ID: {identity}'
    cv2.putText(im, label, (int(bbox_left), int(bbox_top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main(video_path, lane_coordinates_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        model_path = 'yolov5/best.torchscript_gpu.pt'
    else:
        model_path = 'yolov5/best.torchscript_cpu.pt'
    
    names_path = 'yolov5/coco.names'
    
    # get areas from the video
    lane_coordinates = load_lane_coordinates(lane_coordinates_path)
    lane_areas = li.convert_coordinates_to_polygons(lane_coordinates)


    module = torch.jit.load(model_path).to(device).eval()
    class_names = load_names(names_path)

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    base_name = os.path.basename(video_path)
    output_name = os.path.splitext(base_name)[0] + "_tracked.mp4"
    output_path = os.path.join('output_video', output_name)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    id_area = {}
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_function(frame, module, class_names, device, deepsort, lane_areas, id_area)
            video_writer.write(processed_frame)
            pbar.update(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f'Output video saved to {output_path}')


if __name__ == '__main__':
    video_path = 'test_video/test_pool_trimmed_2min.mp4'
    lane_coordinates_path = 'lane_identification/test_pool_trimmed_2min_lane_coordinates.txt'
    main(video_path, lane_coordinates_path)
