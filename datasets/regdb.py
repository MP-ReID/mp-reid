import glob
import re
import os
import os.path as osp
import numpy as np

from .bases import BaseImageDataset
from collections import defaultdict
import random

class RegDB(BaseImageDataset):
    dataset_dir = '/nas_24/RegDB'

    def __init__(self, root='', verbose=True, pid_begin=0, exp_setting='rgb2ir', **kwargs):
        super(RegDB, self).__init__()
        self.dataset_dir = root if root else self.dataset_dir
        self.exp_setting = exp_setting  # 'rgb2ir' 或 'ir2rgb'
        self.pid_begin = pid_begin

        exp_setting = self.exp_setting.split('_')
        self.trial = exp_setting[1]       
        
        # 索引文件路径
        self.train_visible_idx_file = osp.join(self.dataset_dir, 'idx', f'train_visible_{self.trial}.txt')
        self.train_thermal_idx_file = osp.join(self.dataset_dir, 'idx', f'train_thermal_{self.trial}.txt')
        self.test_visible_idx_file = osp.join(self.dataset_dir, 'idx', f'test_visible_{self.trial}.txt')
        self.test_thermal_idx_file = osp.join(self.dataset_dir, 'idx', f'test_thermal_{self.trial}.txt')
        
        # 根据实验设置决定如何处理数据
        train = self._process_train()

        if exp_setting[0] == 'rgb2ir':
            # 可见光作为查询集，红外作为图库集
            query = self._process_query_visible()
            gallery = self._process_gallery_thermal()
        elif exp_setting[0] == 'ir2rgb':
            # 红外作为查询集，可见光作为图库集
            query = self._process_query_thermal()
            gallery = self._process_gallery_visible()
        else:
            raise ValueError(f"Unsupported exp_setting: {self.exp_setting}")
            
        if verbose:
            print(f"=> RegDB loaded, Trial: {self.trial}, Experiment: {self.exp_setting}")
            self.print_dataset_statistics(train, query, gallery)
            
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    
    def _process_train(self):
        """处理训练数据，合并可见光和红外训练图像"""
        # 读取可见光训练索引
        visible_train_data = self._read_idx_file(self.train_visible_idx_file)
        # 读取红外训练索引
        thermal_train_data = self._read_idx_file(self.train_thermal_idx_file)
        
        # 合并数据集
        visible_dataset = self._build_dataset(visible_train_data, cam_id=0)
        thermal_dataset = self._build_dataset(thermal_train_data, cam_id=1)
        
        return visible_dataset + thermal_dataset
    
    def _process_query_visible(self):
        """处理可见光查询集"""
        visible_test_data = self._read_idx_file(self.test_visible_idx_file)
        return self._build_dataset(visible_test_data, cam_id=0, relabel=False)
    
    def _process_gallery_thermal(self):
        """处理红外图库集"""
        thermal_test_data = self._read_idx_file(self.test_thermal_idx_file)
        return self._build_dataset(thermal_test_data, cam_id=1, relabel=False)
    
    def _process_query_thermal(self):
        """处理红外查询集"""
        thermal_test_data = self._read_idx_file(self.test_thermal_idx_file)
        return self._build_dataset(thermal_test_data, cam_id=1, relabel=False)
    
    def _process_gallery_visible(self):
        """处理可见光图库集"""
        visible_test_data = self._read_idx_file(self.test_visible_idx_file)
        return self._build_dataset(visible_test_data, cam_id=0, relabel=False)
    
    def _read_idx_file(self, idx_file_path):
        """
        读取索引文件
        
        索引文件格式:
        图像路径 重新标记的ID
        例如:
        Visible/285/male_back_v_05528_285.bmp 0
        Visible/117/male_front_v_01486_117.bmp 1
        """
        data = []
        with open(idx_file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                
                try:
                    img_path, relabel_id = line.split()
                    relabel_id = int(relabel_id)
                    
                    # 从图像路径中提取原始ID
                    # 例如: Visible/285/male_back_v_05528_285.bmp -> 285
                    try:
                        orig_id = int(img_path.split('/')[1])
                    except:
                        # 如果无法从路径中提取，则尝试从文件名中提取
                        # 例如: male_back_v_05528_285.bmp -> 285
                        orig_id = int(img_path.split('_')[-1].split('.')[0])
                    
                    data.append((img_path, orig_id, relabel_id))
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
                    continue
        
        return data
    
    def _build_dataset(self, data_list, cam_id, relabel=True):
        """
        构建数据集列表
        
        Args:
            data_list: 包含(img_path, orig_id, relabel_id)元组的列表
            cam_id: 相机ID (0:可见光, 1:红外)
            relabel: 是否使用重新标记的ID
            
        Returns:
            dataset: 包含(img_path, pid, camid, trackid)元组的列表
        """
        dataset = []
        
        for img_path, orig_id, relabel_id in data_list:
            full_img_path = osp.join(self.dataset_dir, img_path)
            pid = relabel_id if relabel else orig_id
            dataset.append((full_img_path, self.pid_begin + pid, cam_id, 0))  # trackid=0
        
        return dataset
    
    def combine_all_trials(self, output_dir=None):
        """
        计算所有trial (1-10)的平均性能，用于最终评估
        
        Args:
            output_dir: 如果提供，将把每个trial的性能保存到这个目录下
            
        Returns:
            tuple: 所有metric的平均值和标准差
        """
        results = []
        
        for trial in range(1, 11):
            # 这部分可以与外部评估代码集成，用于自动评估
            # 这里仅提供结构，具体实现需要与评估代码配合
            print(f"Evaluating Trial {trial}...")
            
            # 这里假设有一个外部评估函数evaluate_trial
            # result = evaluate_trial(trial, self.exp_setting)
            # results.append(result)
            
            # 如果提供了输出目录，可以保存单个trial的结果
            # if output_dir:
            #     save_path = osp.join(output_dir, f'trial_{trial}_result.txt')
            #     with open(save_path, 'w') as f:
            #         f.write(str(result))
        
        # 计算平均值和标准差
        # mean_result = np.mean(results, axis=0)
        # std_result = np.std(results, axis=0)
        
        # return mean_result, std_result 