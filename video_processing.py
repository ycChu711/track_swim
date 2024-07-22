import cv2
import torch
import os
import numpy as np
from tqdm import tqdm
import lane_identification.lane_identification as li

from utils import load_names, letterbox_image, post_processing
from tracking import deepsort, process_result, filter_overlapping_detections, fix_offset, get_center_point, update_track_id_and_lane, draw_bounding_box
from lane_utils import load_lane_coordinates

def process_function(im, module, class_names, device, deepsort, areas, id_to_lane_mapping, original_to_current_id_mapping):
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

        filtered_outputs = filter_overlapping_detections(outputs, iou_threshold=0.5)

        if len(filtered_outputs) > 0:
            for j, output in enumerate(filtered_outputs):
                bbox_left, bbox_top, bbox_right, bbox_bottom, class_id, identity = output

                bbox_left, bbox_top, bbox_right, bbox_bottom = fix_offset(bbox_left, bbox_top, bbox_right, bbox_bottom)

                center_x, center_y = get_center_point(bbox_left, bbox_top, bbox_right, bbox_bottom)

                object_area_name = li.assign_objects_to_areas(center_x, center_y, areas)
                
                # print identity type
                identity = int(identity)
                # Update identity based on lane change
                updated_identity = update_track_id_and_lane(identity, object_area_name, id_to_lane_mapping, original_to_current_id_mapping)

                draw_bounding_box(im, bbox_left, bbox_top, bbox_right, bbox_bottom, class_names[class_id], updated_identity, id_to_lane_mapping)

    return im

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
    id_to_lane_mapping = {}
    original_to_current_id_mapping = {}
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_function(frame, module, class_names, device, deepsort, lane_areas, id_to_lane_mapping, original_to_current_id_mapping)
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