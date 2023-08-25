# Wheelock Video Analysis 2023S Update
Our contribution is in deployment folder.

## Contributions of This Semester
For understanding flow of classes from teacher-uploaded classroom recordings with an unsupervised approach, we attempted to deploy a total of three models, two of which worked without problems and were able to produce nontrivial results. The two that are up and running are Slowfast, a motion recognition & object detection pipeline developed by Meta Research, and PDVC, a zero-shot dense video captioning model. Their deployment links with instructions are included below.


## Slowfast
Hugging Face deployment. Upgrade the CPU before running model. <br>
Link: https://huggingface.co/spaces/spark-ds549/SlowFast2023

### How to Use
Open the link on browser, upload a short video and click on Submit. Relevant files in deployment/SlowFast are a replica of those hosted on the Hugging Face Space.

### Output
The output is a video with target bounding boxes with descriptions of the actions predicted.


## PDVC
Google Colab deployment. <br>
Link: https://colab.research.google.com/drive/1MpduTs6l6cSokzJTpEa29R21i4RpHLNx?usp=sharing <br>
Next steps: run PDVC on SCC and Hugging Face.

### How to Use
Put the jupyter notebook on Google Colab and follow the instructions.

### Output
There are two outputs. One output is a video that visualizes the captions, and the other output is a json file containing time stamps with the generated captions.


## VideoBERT

#### This pipeline is depreciated. Attempts have been halted due to high computational power & time cost of training a video transformer. Legacy code is attached as a proof of work. 

### How to Use
Put the video_bert_labeling.ipynb under SCC directory projectnb/sparkgrp/ml-wheelock-video-grp/ and run the python scripts from the notebook

### Desired Output
A video transformer that is capable of categorizing an unseen clip into a few categories it was trained on
