We refer to the research from in the previous semester as background. Here, we look at more papers that are relevant to this semester's work. 

# Paper Review
## Generating Labels from Videos
[Unsupervised Learning from Videos using Temporal Coherency Deep Networks](https://www.sciencedirect.com/science/article/pii/S1077314218301772)
* The authors train a [model](https://github.com/gramuah/unsupervised) to learn from unlabeled videos. They generate features and show how it can be used to discover actions in the videos. 
* We can apply transfer learning on their pre-trained models and use the feature vectors to statistically find labels for segments of video.

[Spatiotemporal Contrastive Video Representation Learning](https://arxiv.org/pdf/2008.03800v4)
* The paper describes a self-supervised contrastive video representation learning (CVRL) method that learns spatiotemporal visual representations from unlabeled videos.
* The method uses a special way of comparing and grouping together similar and different video clips to learn to identify patterns in the videos. The authors also designed some new ways of making the videos more informative to help the method learn better.
* The authors provide a [link](https://github.com/tensorflow/models/tree/master/official/projects/video_ssl) to their model. 

## Video Description Generation
[Video Description Generation using Audio and Visual Cues](https://dl.acm.org/doi/pdf/10.1145/2911996.2912043?casa_token=5NFurzegl80AAAAA:QrSX0W5sBykM8mHWB3uMj_6eFGCk32zUhlZ54mu21-CIPupCRV75hPw5SSMoFn5zIVNxOfncQmo)
* The paper talks about generating natural language descriptions on video clips based on both visual and audio cues using LSTMs and CNNs. 
* We can use this method to classify videos into certain scenarios using NLP models, and apply different pipelines based on scenario (classroom_front_view, classroom_side_view, zoom, etc.). 

## Improving Image Resolution
[Fourier Image Transformer](https://arxiv.org/abs/2104.02555)
* The paper proposes a novel technique using the Fourier Transform to generate image representations for deep learning applications, which is useful for making images have higher resolution.
* The authors train a [model](https://github.com/juglab/FourierImageTransformer) to use a sequential image representation describes the whole image at reduced resolution. 
* We can also use this approach to analyze images in Fourier space.

## Architecture
[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
* This paper introduces a CNN that significantly improves performance compared to previous architectures.
* The authors provide a [link](https://github.com/facebookresearch/ConvNeXt) to their implementation. 