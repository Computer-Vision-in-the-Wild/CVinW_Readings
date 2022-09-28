# CVinW Readings [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings)

``[Computer Vision in the Wild (CVinW)](https://computer-vision-in-the-wild.github.io/eccv-2022/)'' is an emerging field. This writeup provides a quick introduction of CVinW and maintains a collection of papers on the topic. If you find some missing papers or resourses, please open issues or pull requests (recommended).


# Table of Contents

- [What is Computer Vision in the Wild (CVinW)?](#what-is-computer-vision-in-the-wild)
  - [Goals of CVinW](#star-goals-of-cvinw)
  - [Task Transfer Scenarios are Broad](#one-task-transfer-scenarios-are-broad) 
  - [Task Transfer Cost is Low](#two-task-transfer-cost-is-low )
  - [Benchmarks](#cinema-benchmarks)
- [Papers on Task-level Transfer with Pre-trained Models](#papers-on-task-level-transfer-with-pre-trained-models)
  - [Image Classification in the Wild](#image-classification-in-the-wild)
  - [Object Detection in the Wild](#object-detection-in-the-wild)
  - [Segmentation in the Wild](#segmentation-in-the-wild)
  - [Video Classification in the Wild](#video-classification-in-the-wild)
  - [Others](#others-visual-recognition-in-the-wild)
- [Papers on Efficient Model Adaptation](#papers-on-efficient-model-adaptation)  
  - [Parameter-Efficient Methods](#parameter-efficient-methods)
- [Acknowledgements](#acknowledgements)


# What is Computer Vision in the Wild?

### :star: Goals of CVinW
Developing a transferable foundation model/system that can *effortlessly* adapt to *a large range of visual tasks* in the wild. It comes with two key factors: (i) The task transfer scenarios are broad, and (ii) The task transfer cost is low. The main idea is illustrated as follows, please see the detailed description in [ELEVATER paper](https://arxiv.org/abs/2204.08790). 

### :one: Task Transfer Scenarios are Broad

We illustrate and compare CVinW with other settings using a 2D chart in Figure 1, where the space is constructed with two orthogonal dimensions:
input image distribution and output concept set. The 2D chart is divided into four quadrants, based on how the model evaluation stage is different from model development stage. For any visual recognition problems at different granularity such as image classification, object detection and segmentation, the modeling setup cann be categorized into one of the four settings. We see an emerging trend on moving towards CVinW. Interested in the various pre-trained vision models that move towards CVinW? please check out Section [``Papers on Task-level Transfer with Pre-trained Models''](#papers-on-task-level-transfer-with-pre-trained-models).

<table>
<tr>
  <td  width="50%">
<ul>
  <li><b>The Close-Set Setting. </b> Both training and evaluation distributions are consistent in both dimensions, a typical setting in ML/CV textbooks.</li>
  <li><b>Open-Set/Vocabulary/World Setting.</b> It allows new concepts in evaluation, while typically remains the same visual domain. Please see examples in <a href='https://arxiv.org/abs/1707.00600'>image classification</a>  and <a href='https://arxiv.org/abs/2011.10678'>object detection</a>. </li>
  <li><b>Domain Generalization Setting.</b> Domain shift allows new visual domain in evaluation, while typically remains the same concept pool. Please see examples such as <a href='https://arxiv.org/abs/2007.01434'>DomainBed</a>  and <a href='http://ai.bu.edu/M3SDA/'>DomainNet</a>.  </li>
  <li style="background-color:powderblue;"><b>Computer Vision in the Wild Setting. </b> CVinW allows the flexibility in both dimensions, where any new tasks/datasets in the wild essentially fall into.</li>
</ul>    

</td>
<td>
    <img src="images/fig_cvinw.png" style="width:100%;"> 
</td>
</tr> 
<tr>
  <th> A brief definition with a four-quadrant chart </th>
  <th>Figure 1: The comparison of CVinW with other existing settings</th>
</tr>
</table>


### :two: Task Transfer Cost is Low

One major advantage of pre-trained models is the promise that they can transfer to downstream tasks *effortlessly*. The model adaptation cost is considered in two orthogonal dimensions: *sample-efficiency* and *parameter-efficiency*, as illustrated in Figure 2.  The bottom-left corner and  top-right corner is the most inexpensive and  expensive adaptation strategy, respectively. One may interpolate and  make combinations in the 2D space, to get different model adaptation methods with different cost. To efficient adapt large vision models of the gradaully increaseing size, we see an emerging need on efficient model adaptation. Interested in contributing your smart efficient adaptation algorithms and see how it differs from existing papers? please check out Section [``Papers on Efficient Model Adaptation''](#papers-on-efficient-model-adaptation)  .

<table>
<tr>
  <td  width="50%">
<ul>
  <li><b>Sample-efficiency: Zero-, Few-, and Full-shot. </b> Due to the high cost of annotating data, it is often desired to provide a small number of labeled image-label pairs in downstream datasets. Transferable models should be able to reach high performance in this data-limited scenario..</li>
  <li><b>Parameter-efficiency: Frozen Model Inference, Prompting Tuning, Linear Probing vs Full Model Fine-tuning..</b> A smaller number of trainable parameter in model adaptation typically means a small training cost in a new task. </li>
</ul>    

</td>
<td>
    <img src="images/fig_adapation_cost.png" style="width:100%;">
</td>
</tr> 
<tr>
  <th> A breakdown definition of efficient model adaptation</th>
  <th>Figure 2: The 2D chart of model adaptation cost.</th>
</tr>
</table>

 
###  :cinema: Benchmarks

<p>
<font size=3><b>ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models.</b></font>
<br>
<font size=2>Chunyuan Li*, Haotian Liu*, Liunian Harold Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Yong Jae Lee, Houdong Hu, Zicheng Liu, Jianfeng Gao.</font>
<br>
<font size=2> NeurIPS 2022 (Datasets and Benchmarks Track).</font>
<a href='https://arxiv.org/abs/2204.08790'>[paper]</a> <a href='https://computer-vision-in-the-wild.github.io/ELEVATER/'>[benchmark]</a>    
</p>


# Papers on Task-level Transfer with Pre-trained Models

## Image Classification in the Wild

<p>
<font size=3><b>Learning Transferable Visual Models From Natural Language Supervision.</b></font>
<br>
<font size=2>Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.</font>
<br>
<font size=2>ICML 2021.</font>
<a href='https://arxiv.org/abs/2103.00020'>[paper]</a> <a href='https://github.com/OpenAI/CLIP'>[code]</a>    
</p>

<p>
<font size=3><b>Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision.</b></font>
<br>
<font size=2>Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.</font>
<br>
<font size=2>ICML 2021.</font>
<a href='https://arxiv.org/abs/2102.05918'>[paper]</a> 
</p>



## Object Detection in the Wild
<p>
<font size=3><b>[VILD] 
Open-vocabulary Object Detection via Vision and Language Knowledge Distillation.
</b></font>
<br>
<font size=2>Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, Yin Cui.</font>
<br>
<font size=2>ICLR 2022.</font>
<a href='https://arxiv.org/abs/2104.13921'>[paper]</a>
</p>


<p>
<font size=3><b>GLIP: Grounded Language-Image Pre-training.</b></font>
<br>
<font size=2>Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, Jianfeng Gao</font>
<br>
<font size=2>CVPR 2022.</font>
<a href='https://arxiv.org/abs/2112.03857'>[paper]</a> <a href='https://github.com/microsoft/GLIP'>[code]</a>    
</p>

<p>
<font size=3><b>RegionCLIP: Region-based Language-Image Pretraining.</b></font>
<br>
<font size=2>Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, Jianfeng Gao.</font>
<br>
<font size=2>CVPR 2022.</font>
<a href='https://arxiv.org/abs/2112.09106'>[paper]</a> <a href='https://github.com/microsoft/RegionCLIP'>[code]</a>    
</p>

<p>
<font size=3><b>MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding.</b></font>
<br>
<font size=2>Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, Nicolas Carion.</font>
<br>
<font size=2>ICCV 2021.</font>
<a href='https://arxiv.org/abs/2104.12763'>[paper]</a> <a href='https://github.com/ashkamath/mdetr'>[code]</a>    
</p>

<p>
<font size=3><b>Open-Vocabulary DETR with Conditional Matching Yuhang.</b></font>
<br>
<font size=2>Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, Chen Change Loy.</font>
<br>
<font size=2>ECCV 2022.</font>
<a href='https://arxiv.org/abs/2203.11876'>[paper]</a> <a href='https://github.com/yuhangzang/ov-detr'>[code]</a>    
</p>



<p>
<font size=3><b>OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers.</b></font>
<br>
<font size=2>Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, Neil Houlsby.</font>
<br>
<font size=2>ECCV 2022.</font>
<a href='https://arxiv.org/abs/2205.06230'>[paper]</a>     
</p>


<p>
<font size=3><b>Detic: Detecting Twenty-thousand Classes using Image-level Supervision.</b></font>
<br>
<font size=2>Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, Ishan Misra.</font>
<br>
<font size=2>ECCV 2022.</font>
<a href='https://arxiv.org/abs/2201.02605'>[paper]</a> <a href='https://github.com/facebookresearch/Detic/'>[code]</a>        
</p>


<p>
<font size=3><b>X-DETR: A Versatile Architecture for Instance-wise Vision-Language Tasks.</b></font>
<br>
<font size=2>Zhaowei Cai, Gukyeong Kwon, Avinash Ravichandran, Erhan Bas, Zhuowen Tu, Rahul Bhotika, Stefano Soatto.</font>
<br>
<font size=2>ECCV 2022.</font>
<a href='https://arxiv.org/abs/2204.05626'>[paper]</a>     
</p>

<p>
<font size=3><b>PromptDet: Towards Open-vocabulary Detection using Uncurated Images.</b></font>
<br>
<font size=2>Chengjian Feng, Yujie Zhong, Zequn Jie, Xiangxiang Chu, Haibing Ren, Xiaolin Wei, Weidi Xie, Lin Ma.</font>
<br>
<font size=2>ECCV 2022.</font>
<a href='https://arxiv.org/abs/2203.16513'>[paper]</a> <a href='https://fcjian.github.io/promptdet/'>[code]</a>        
</p>

<p>
<font size=3><b>Class-agnostic Object Detection with Multi-modal Transformer.</b></font>
<br>
<font size=2>Muhammad Maaz, Hanoona Rasheed, Salman Khan, Fahad Shahbaz Khan, Rao Muhammad Anwer and Ming-Hsuan Yang.</font>
<br>
<font size=2>ECCV 2022.</font>
<a href='https://arxiv.org/abs/2111.11430'>[paper]</a> <a href='https://github.com/mmaaz60/mvits_for_class_agnostic_od'>[code]</a>    
</p>

<p>
<font size=3><b>[Object Centric OVD] -- Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection.</b></font>
<br>
<font size=2>Hanoona Rasheed, Muhammad Maaz, Muhammad Uzair Khattak, Salman Khan, Fahad Shahbaz Khan.</font>
<br>
<font size=2>NeurIPS 2022</font>
<a href='https://arxiv.org/abs/2207.03482'>[paper]</a> <a href='https://github.com/hanoonaR/object-centric-ovd'>[code]</a>    
</p>

<p>
<font size=3><b>GLIPv2: Unifying Localization and Vision-Language Understanding.</b></font>
<br>
<font size=2>Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, Jianfeng Gao.</font>
<br>
<font size=2>NeurIPS 2022</font>
<a href='https://arxiv.org/abs/2206.05836'>[paper]</a> <a href='https://github.com/microsoft/GLIP'>[code]</a>    
</p>

<p>
<font size=3><b>Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone.</b></font>
<br>
<font size=2>Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng Liu, Ce Liu, Yann LeCun, Nanyun Peng, Jianfeng Gao, Lijuan Wang.</font>
<br>
<font size=2>NeurIPS 2022</font>
<a href='https://arxiv.org/abs/2206.07643'>[paper]</a> <a href='https://github.com/microsoft/FIBER'>[code]</a>    
</p>

<p>
<font size=3><b>DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection.</b></font>
<br>
<font size=2> Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei zhang, Zhenguo Li, Chunjing Xu, Hang Xu.</font>
<br>
<font size=2>NeurIPS 2022</font>
<a href='https://arxiv.org/abs/2209.09407v1'>[paper]</a>  
</p>

<p>
<font size=3><b>OmDet: Language-Aware Object Detection with Large-scale Vision-Language Multi-dataset Pre-training.</b></font>
<br>
<font size=2>Tiancheng Zhao, Peng Liu, Xiaopeng Lu, Kyusong Lee</font>
<br>
<font size=2>arxiv:2209.0594</font>
<a href='https://arxiv.org/abs/2209.05946'>[paper]</a> 
</p>


## Segmentation in the Wild

## Video Classification in the Wild

## Others Visual Recognition in the Wild

# Papers on Efficient Model Adaptation

## Parameter-Efficient Methods

# Acknowledgements

We thank all the authors above for their great works! Related Reading List 

- [[Awesome Detection Transformer]](https://github.com/IDEACVR/awesome-detection-transformer) 
