# MP-ReID
**Authors: Ruiyang Ha,Songyi Jiang,Bin Li,Bikang Pan,Yihang Zhu,Junjie Zhang,Xiatian Zhu,Shaogang Gong,Jingya Wang**

**Paper:** https://arxiv.org/abs/2503.17096

## Abstract
Conventional person re-identification (ReID) research is often limited to single-modality sensor data from static cameras, which fails to address the complexities of real-world scenarios where multi-modal signals are increasingly prevalent. For instance, consider an urban ReID system integrating stationary RGB cameras, nighttime infrared sensors, and UAVs equipped with dynamic tracking capabilities. Such systems face significant challenges due to variations in camera perspectives, lighting conditions, and sensor modalities, hindering effective person ReID. To address these challenges, we introduce the MP-ReID benchmark, a novel dataset designed specifically for multi-modality and multi-platform ReID. This benchmark uniquely compiles data from 1,930 identities across diverse modalities, including RGB, infrared, and thermal imaging, captured by both UAVs and ground-based cameras in indoor and outdoor environments. Building on this benchmark, we introduce Uni-Prompt ReID, a framework with specific-designed prompts, tailored for cross-modality and cross-platform scenarios. Our method consistently outperforms state-of-the-art approaches, establishing a robust foundation for future research in complex and dynamic ReID environments.

## Dataset Download
You can see more details in [dataset](dataset.md) page.
Please fill in the <a href="https://drive.google.com/file/d/1hImLEMcsBB2kNV4McGyksVAumLjZQoUU/view?usp=sharing">agreement</a> and send it to vihumanlab@gmail.com to get the MP-ReID Dataset.


### Installation

```
conda create -n UniPrompt python=3.8
conda activate UniPrompt
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
### Training

For example, if you want to train the baseline in cctv_ir_cctv_rgb set, you need to modify the bottom of configs/ours/cctv_ir_cctv_rgb.yml to

```
DATASETS:
   NAMES: ('mmmp')
   ROOT_DIR: ('your_dataset_dir')
   EXP_SETTING: ('exp_cctv_ir_cctv_rgb')
OUTPUT_DIR: 'your_output_dir'
```

then run 

```
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/ours/cctv_ir_cctv_rgb.yml
```

We are also providing additional files that might be used during the training phase. Please refer to the link in the [Dataset ](dataset.md) section for more details. 
## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{ha2025multi,
            title={Multi-modal Multi-platform Person Re-Identification: Benchmark and Method},
             author={Ha, Ruiyang and Jiang, Songyi and Li, Bin and Pan, Bikang and Zhu, Yihang and Zhang, Junjie and Zhu, Xiatian and Gong, Shaogang and Wang, Jingya},
             journal={arXiv preprint arXiv:2503.17096},
            year={2025}
          }
```
<!--
**MP-ReID/mp-reid** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
