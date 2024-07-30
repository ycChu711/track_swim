import cv2
import numpy as np
import torch

def load_names(path):
    '''
    Description: Load class names from a file
    Arguments:
        path: str, path to the file containing class names
    Returns:
        class_names: list, list of class names    
    '''
    class_names = []
    with open(path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def letterbox_image(src, out_size):
    '''
    Description:
    Resize the image to the output size while maintaining the aspect ratio.
    Pad the resized image to match the output size.
    Arguments:
        src: np.array, input image
        out_size: tuple, output size (width, height)
    Returns:
        dst: np.array, resized and padded image
        [left, top, scale]: list, padding information
    '''
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
    '''
    Description:
    Scale the bounding box coordinates back to the original image size
    Arguments:
        data: list, list of dictionaries containing bounding box coordinates
        pad_w: int, padding width
        pad_h: int, padding height
        scale: float, scaling factor
        img_shape: tuple, original image size (height, width)
    '''
    def clip(n, lower, upper):
        return max(lower, min(n, upper))

    for i in data:
        # Remove the offset
        class_idx = i['class_idx']
        offset = class_idx * 4096
        x1 = i['bbox'][0] - offset
        y1 = i['bbox'][1] - offset
        x2 = i['bbox'][2] - offset
        y2 = i['bbox'][3] - offset

        # Scale and adjust the coordinates
        x1 = (x1 - pad_w) / scale  # x padding
        y1 = (y1 - pad_h) / scale  # y padding
        x2 = (x2 - pad_w) / scale  # x padding
        y2 = (y2 - pad_h) / scale  # y padding

        # Clamp the coordinates
        x1 = clip(x1, 0, img_shape[1])
        y1 = clip(y1, 0, img_shape[0])
        x2 = clip(x2, 0, img_shape[1])
        y2 = clip(y2, 0, img_shape[0])

        i['bbox'] = [x1, y1, x2, y2]


def tensor2detection(offset_boxes, det):
    '''
    Description:
    Convert the tensor output to a list of bounding box coordinates and scores
    Arguments:
        offset_boxes: torch.Tensor, tensor containing the bounding box coordinates
        det: torch.Tensor, tensor containing the detection results
    Returns:
        offset_box_vec: list, list of bounding box coordinates
        score_vec: list, list of detection scores
    '''
    offset_box_vec, score_vec = [], []
    for i in range(offset_boxes.size(0)):
        offset_box_vec.append([int(offset_boxes[i, j]) for j in range(4)])
        score_vec.append(float(det[i, 4]))
    return offset_box_vec, score_vec


def post_processing(detections, pad_w, pad_h, scale, img_shape, conf_thres, iou_thres):
    '''
    Description:
    Perform non-maximum suppression on the detection results
    Arguments:
        detections: torch.Tensor, tensor containing the detection results
        pad_w: int, padding width
        pad_h: int, padding height
        scale: float, scaling factor
        img_shape: tuple, original image size (height, width)
        conf_thres: float, confidence threshold
        iou_thres: float, IoU threshold
    Returns:
        result: list, list of dictionaries containing bounding box coordinates, scores, and class indices
    '''
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
    indices = cv2.dnn.NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres)
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

def xywh2xyxy(x):
    '''
    Description:
    Convert bounding box coordinates from (x, y, w, h) to (x1, y1, x2, y2)
    Arguments:
        x: torch.Tensor, tensor containing the bounding box coordinates
    Returns:
        y: torch.Tensor, tensor containing the converted bounding box coordinates
    '''
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y