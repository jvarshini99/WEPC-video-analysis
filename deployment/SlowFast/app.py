import gradio as gr
import cv2
import os
import numpy as np
import pandas as pd

from run import slow_fast_train  # Import the slow_fast_train function from your provided code


try:
    import detectron2
except:
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

def predict(input_video, frames):
    print("START!!!")
    
    # Save the uploaded video to a temporary file
    # input_video_path = 'input_video.mp4'
    # with open(input_video_path, 'wb') as f:
    #     f.write(input_video.read())
    
    # Perform SlowFast inference
    video_1, metadata_1 = slow_fast_train(input_video)  # Call the slow_fast_train function

    return video_1

iface = gr.Interface(
    predict,
    inputs=[gr.Video(),gr.Slider(1, 100, value=15)],
    outputs=[gr.Video()],
    title="Video Analysis"
)

iface.launch(show_error=True, debug=True)
