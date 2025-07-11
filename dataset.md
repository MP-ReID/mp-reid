# MP-ReID: A Multi-Modal, Multi-Platform Dataset for Person Re-Identification
![Header](teaser.png)

MP-ReID stands as the inaugural dataset specifically constructed for research in multi-modal and multi-platform person re-identification. Data collection for this dataset employed a dual-platform approach, utilizing both ground-based surveillance and unmanned aerial vehicles (UAVs). Furthermore, the dataset incorporates three distinct modalities: RGB, thermal infrared, and near-infrared.

The creation of this comprehensive dataset spanned over four months of data acquisition and includes 1,930 unique person identities from 14 camera views with 136,156 bounding boxes.

MP-ReID encompasses a wealth of diverse pedestrian information, offering an invaluable resource for advancing research in multi-modal and multi-platform person re-identification.

## Data structure
The dataset has the following folder structure which will either be produced by the download or generation scripts.
```text
MP-ReID/
├── 1/
│ ├── 0001/
│ │ ├── 0001.jpg
│ │ ├── 0002.jpg
│ │ ├── ...
│ │ ├── 0058.jpg
│ ├── 0002/
│ ├── 0003/
│ ├── ...
│ ├── 0568/
├── 2/
│ ├── 0001/
│ ├── 0002/
│ ├── 0003/
│ ├── ...
│ ├── 0747/
...
├── 14/
│ ├── 0001/
│ ├── 0002/
│ ├── 0003/
│ ├── ...
│ ├── 1268/
```
The MP-ReID dataset is captured by 14 cameras, with each camera corresponding to a folder. Cameras 01-06 are ground-based RGB cameras, among which 01 and 03 are set up indoors, and the rest are set up outdoors. Cameras 07-12 are ground-based IR cameras, with 07 and 09 set up indoors, and the rest are set up outdoors. Camera 13 is a drone-based RGB camera, while camera 14 is a drone-based thermal camera.
## Data Split
Following the proposed experimental benchmarks, experiments are categorized into six distinct types. For each experiment type, separate training and testing sets have been designated. Pre-partitioned .txt files and a partitioning script are provided, allowing for custom re-partitioning if needed. You can find specific experimental divisions are recorded in the `split` folder. For example, exp_cctv_ir_cctv_rgb represents the experiment from infrared images collected by CCTV to the RGB images collected by CCTV. And you can find the script for spliting in `scripts` folder.
You can find the spliting files like the following structure.
```
split/
├── exp_cctv_ir_cctv_rgb/
│ ├── available_id.txt
│ ├── test_id.txt
│ ├── train_id.txt
│ ├── val_id.txt
├── exp_cctv_ir_uav_ir
│ ├── available_id.txt
│ ├── test_id.txt
│ ├── train_id.txt
│ ├── val_id.txt
...
├── exp_uav_ir_uav_rgb
│ ├── available_id.txt
│ ├── test_id.txt
│ ├── train_id.txt
│ ├── val_id.txt
```

## Dataset download
Please fill in the <a href="https://drive.google.com/file/d/1hImLEMcsBB2kNV4McGyksVAumLjZQoUU/view?usp=sharing">agreement</a> and send it to vihumanlab@gmail.com to get the MP-ReID Dataset.

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


