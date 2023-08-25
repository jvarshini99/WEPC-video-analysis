import numpy as np
import pandas as pd
import cv2
import torch
import warnings
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import ffmpeg
import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection # Another option is slow_r50_detection
from visualization import VideoVisualizer

import time

import subprocess
import tempfile

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes


def ava_inference_transform(
    clip,
    boxes,
    num_frames = 32, # 4 if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, # if using slowfast_r50_detection, change None to 4
    device = 'cpu'): 

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes)

    # Normalize images by mean and std.
    clip = normalize(clip, np.array(data_mean, dtype=np.float32), np.array(data_std, dtype=np.float32))

    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(clip, 1, torch.linspace(
            0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway.unsqueeze(0).to(device), fast_pathway.unsqueeze(0).to(device)]

    return clip, torch.from_numpy(boxes), ori_boxes

# get video info
def with_opencv(filename):
    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    s = round(frame_count / fps)
    video.release()
    return int(s), fps

top_k = 1
video_model = slowfast_r50_detection(True) # Another option is slow_r50_detection(True) 
video_model = video_model.eval().to(device)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = device
predictor = DefaultPredictor(cfg)
# Create an id to label name mapping
label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
# Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
video_visualizer = VideoVisualizer(81, label_map, top_k=top_k, mode="thres",thres=0.5) #get top3 predictions show in each bounding box

def slow_fast_train(file_path):
    start_time = time.time()
    
    #preprocess video data
    encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(file_path)

    # Video predictions are generated each frame/second for the wholevideo.
    total_sec, fps = with_opencv(file_path)
    time_stamp_range = range(0, total_sec) # time stamps in video for which clip is sampled
    clip_duration = 1.0 # Duration of clip used for each inference step.
    gif_imgs = []
    xleft, ytop, xright, ybottom = [], [], [], []
    labels = []
    time_frame = []
    scores = []

    initial_time = time.time() - start_time
    print(f"slow_fast time used to import model: {initial_time:.2f} seconds")
    start_time = time.time()

    print(time_stamp_range)
    for time_stamp in time_stamp_range:

        # Generate clip around the designated time stamps
        inp_imgs = encoded_vid.get_clip(
            time_stamp - clip_duration/2.0,
            time_stamp + clip_duration/2.0)  
        inp_imgs = inp_imgs['video']

        #if time_stamp % 15 == 0:
            # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
            # We use the the middle image in each clip to generate the bounding boxes.
        inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
        inp_img = inp_img.permute(1,2,0)
    
        # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
        predicted_boxes = get_person_bboxes(inp_img, predictor)
        if len(predicted_boxes) == 0:
            print("Skipping clip no frames detected at time stamp: ", time_stamp)
            continue
        
        # Preprocess clip and bounding boxes for video action recognition.
        inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy(), device=device)
        # Prepend data sample id for each bounding box.
        # For more details refere to the RoIAlign in Detectron2
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
        
        # Generate actions predictions for the bounding boxes in the clip.
        # The model here takes in the pre-processed video clip and the detected bounding boxes.
        preds = video_model(inputs, inp_boxes.to(device)) #change inputs to inputs.unsqueeze(0).to(device) if using slow_r50
        
        preds = preds.to('cpu')
        # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
        preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)

        # Plot predictions on the video and save for later visualization.
        inp_imgs = inp_imgs.permute(1,2,3,0)
        inp_imgs = inp_imgs/255.0
        out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
        gif_imgs += out_img_pred

        #format of bboxes(x_left, y_top, x_right, y_bottom)
        predicted_boxes_lst = predicted_boxes.tolist()
        topscores, topclasses = torch.topk(preds, k=1)
        topscores, topclasses = topscores.tolist(), topclasses.tolist()
        topclasses = np.concatenate(topclasses)
        topscores = np.concatenate(topscores)
                 
        #add top 1 prediction of behaviors in each time step
        for i in range(len(predicted_boxes_lst)):
            xleft.append(predicted_boxes_lst[i][0])
            ytop.append(predicted_boxes_lst[i][1])
            xright.append(predicted_boxes_lst[i][2])
            ybottom.append(predicted_boxes_lst[i][3])
            labels.append(label_map.get(topclasses[i]))
            time_frame.append(time_stamp)
            scores.append(topscores[i])

    print("Finished generating predictions.")
    predict_time = time.time() - start_time
    print(f"slow_fast time used to predict: {predict_time:.2f} seconds")
    start_time = time.time()
    # Generate Metadata file
    metadata = pd.DataFrame()
    metadata['frame'] = time_frame
    metadata['x_left'] = xleft
    metadata['y_top'] = ytop
    metadata['x_right'] = xright
    metadata['y_bottom'] = ybottom
    metadata['label'] = labels
    metadata['confidence'] = scores
            
    height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]

    video_save_path = 'activity_recognition.mp4'
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        video = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
        
        for image in gif_imgs:
            img = (255*image).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.write(img)
        video.release()

        subprocess.run(f"ffmpeg -y -loglevel quiet -stats -i {temp_file.name} -c:v libx264 {video_save_path}".split())

    save_time = time.time() - start_time
    print(f"slow_fast time used to save result: {save_time:.2f} seconds")

    return video_save_path, metadata