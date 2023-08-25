# Wheelock Video Analysis

## Yicun Wu, Dingyuan Xu, Anming Gu, Jeya Varshini Bharath,  2023-02-14 v0.0.1-dev_


## Overview

This project seeks to analyze the use of teaching time based on video data collected from classroom observations.

By accessing a video library of more than 10,000 videos, the team attempts to  eventually answer the following two questions: 
#### 1. How much "idle time" for students exists within the video
#### 2. What is the shape of the lesson based on the type of talk that is occurring at different points in the video

Because the dataset lacked labels for “idle time” the project goals were revised in the Fall 2022 term. Since idle time implies lack of activity, the project was redirected to detect and recognize activities and scene content with the hypothesis that it might be possible to surmise global scene attributes from the combination of constituent recognition results.

The goal for this term is to continue building scene and image understanding capabilities to paint a fuller picture of video content and timeline.



### A. Problem Statement: 
Based on current dataset of classroom recordings during class time, the team attempts to build on last semester's progress in providing insights into student activities during class time. Specifically, we attempt to define and identify "idle time" of students.



### B. Checklist for project completion

1. Work with WEPC to define possible "chunks of time" that might be observed (e.g., direct instruction, small group instruction, independent work time, etc)
2. Create an approach to video analysis that tags segments of video against those possible categories to derive a profile of time use during that observation
3. Create archetypes based on the most common lesson shapes
4. Understand how these archetypes apply with other known variables about classroom (e.g., subject area, % English Learners in the class, etc)
5. Evaluate other pre-trained models for new activity recognition.
6. Explore how to aggregate recognition results to establish high level metadata and descriptions of the video on a timeline.


### C. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI. 
A human can watch all of the videos and manually classify time intervals into "chunks of time" that might be observed. 



### D. Outline a path to operationalization.
Host the project on a web server so anyone without a software engineering background can use it. The end framework is an application that allows input of a video, and outputs details about the observed chunks of time. 

Relevant technologies include a frontend and backend server. 


<!-- _Data Science Projects should have an operationalized end point in mind from the onset. Briefly describe how you see the tool
 produced by this project being used by the end user beyond a jupyter notebook or proof of concept. If possible, be specific and
 call out the relevant technologies_ -->


## Resources


### Data Sets


*   Provided by the WPEC / Teach Forward
    * A library of more than 10,000 videos of classroom observations with corresponding information such as the number of students, subject, and grade.

### References


1. [Analyzing Use of Time in Videos of Classroom Observations ](https://docs.google.com/document/d/1RIr3qk9nLTrtQMpJhb42wpj3Cc0PBM7sXxjcqEZzx34/edit)
2. [Classroom Learning Status Assessment Based on Deep Learning](https://www.hindawi.com/journals/mpe/2022/7049458/)
3. [Audio processing to Mel Spectograms](https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505)

## Weekly Meeting Updates

Meeting notes are kept in the same google doc linked above.