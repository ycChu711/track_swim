import cv2
import numpy as np
import torch

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