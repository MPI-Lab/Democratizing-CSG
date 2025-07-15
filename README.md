<div align="center">

<h1> Democratizing High-Fidelity Co-Speech Gesture Video Generation </h1>

###  <p align="center">Xu Yang<sup>1*‡</sup>, Shaoli Huang<sup>2*</sup>, Shenbo Xie<sup>1*</sup>, Xuelin Chen<sup>2</sup>,</p>
###  <p align="center">Yifei Liu<sup>1</sup>, Changxing Ding<sup>1†</sup></p>
<p align="center">1 South China University of Technology&nbsp;&nbsp;&nbsp;2 Tencent AI Lab</p>
<p align="center">*Equal Contribution.&nbsp;&nbsp;&nbsp;†Corresponding Author.&nbsp;&nbsp;&nbsp;‡Part of his work was done during an internship at Tencent AI Lab.</p>



<a href='https://mpi-lab.github.io/Democratizing-CSG/'>
  <img src='https://img.shields.io/badge/Project-Page-green' width="100">
</a> &nbsp;&nbsp;
<a href='https://arxiv.org/pdf/2507.06812'>
  <img src='https://img.shields.io/badge/Technique-Report-red' width="130">
</a>


## Introduction
<div align="center">
<img width="800" alt="image" src="assets/Teaser_3_8_AI.jpg?raw=true">
</div>

**Abstract:** Co-speech gesture video generation aims to synthesize realistic, audio-aligned videos of speakers, complete with synchronized facial expressions and body gestures. This task presents challenges due to the significant one-to-many mapping between audio and visual content, further complicated by the scarcity of large-scale public datasets and high computational demands. We propose a lightweight framework that utilizes 2D full-body skeletons as an efficient auxiliary condition to bridge audio signals with visual outputs. Our approach introduces a diffusion model conditioned on fine-grained audio segments and a skeleton extracted from the speaker's reference image, predicting skeletal motions through skeleton-audio feature fusion to ensure strict audio coordination and body shape consistency. The generated skeletons are then fed into an off-the-shelf human video generation model with the speaker's reference image to synthesize high-fidelity videos. To democratize research, we present CSG-405—the first public dataset with 405 hours of high-resolution videos across 71 speech types, annotated with 2D skeletons and diverse speaker demographics. Experiments show that our method exceeds state-of-the-art approaches in visual quality and synchronization while generalizing across speakers and contexts. Code, models, and CSG-405 will be publicly released.


## Framework

<div align="center">
<img width="800" alt="image" src="assets/Model_3.8_AI.jpg?raw=true">
</div>

Overview of our co-speech gesture video generation framework. We concatenate the 2D skeleton of the reference image **$R$** with the noisy skeleton sequence **$x_T$** along the frame dimension, providing the body shape cue of the speaker. We then concatenate the embeddings of skeletons and those of audio segments along the feature dimension as the input of the diffusion model, enforcing strict temporal synchronization. Finally, we employ one off-the-shelf human video generation model to produce the co-speech gesture video **$V$** with the synthesized skeleton sequence as an auxiliary condition.

## Dataset
<div align="center">
<img width="800" alt="image" src="assets/Table_dataset_construction.png?raw=true">
</div>
<div align="center">
<img width="800" alt="image" src="assets/Figure_3_17_ages_emotions_gender_races_fixed.jpg?raw=true">
</div>

More details of CSG-405. (a) The proportion of clips for each speech type. (b) Attribute distribution in gender, ethnicity, age, and emotion.
<div align="center">
<img width="800" alt="image" src="assets/Figure_3_17_data_filter_AI.jpg?raw=true">
</div>
Overview of our data collection pipeline. It incorporates four stages including raw video crawling, skeleton annotation, quality control, and post-processing.

## TODO
- [ ] Release code (coming soon)

<!-- ## Visualization

### English

### Chinese

### Singing

### AI-generated Portraits -->



## Citation
If you find this project useful in your research, please consider the citation:

```BibTeX
@inproceedings{yang2025demo,
  title={Democratizing High-Fidelity Co-Speech Gesture Video Generation},
  author={Xu Yang and Shaoli Huang and Shenbo Xie and Xuelin Chen and Yifei Liu and Changxing Ding},
  booktitle={Proceedings of the 2025 International Conference on Computer Vision(ICCV)},
  year={2025}
}
```
