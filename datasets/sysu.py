import glob
import re
import os
import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import random

class SYSU(BaseImageDataset):
    dataset_dir = '/nas_24/sysu'

    def __init__(self, root='', verbose=True, pid_begin=0, exp_setting=None, **kwargs):
        super(SYSU, self).__init__()
        self.dataset_dir = root if root else self.dataset_dir
        self.exp_setting = exp_setting
        self.pid_begin = pid_begin
        
        # 定义SYSU的相机ID
        self.rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']  # RGB相机
        self.ir_cameras = ['cam3', 'cam6']   # IR相机
        self.all_cameras = self.rgb_cameras + self.ir_cameras
        
        # 相机名称到ID的映射
        self.cam_name_to_id = {name: i + 1 for i, name in enumerate(self.all_cameras)}
        
        # ID文件路径
        self.train_id_file = osp.join(self.dataset_dir, 'exp/train_id.txt')
        self.val_id_file = osp.join(self.dataset_dir, 'exp/val_id.txt')
        self.test_id_file = osp.join(self.dataset_dir, 'exp/test_id.txt')
        
        # 根据实验设置决定如何处理数据
        if exp_setting == 'all_train_rgb2ir':
            # 全部训练数据，RGB查询，IR图库
            train = self._process_train(relabel=True)
            query = self._process_query_rgb()
            gallery = self._process_gallery_ir()
        elif exp_setting == 'all_train_ir2rgb':
            # 全部训练数据，IR查询，RGB图库
            train = self._process_train(relabel=True)
            query = self._process_query_ir()
            gallery = self._process_gallery_rgb()
        else:
            print(f"不支持的实验设置: {exp_setting}")
            train, query, gallery = [], [], []
            
        if verbose:
            print(f"=> SYSU 数据集加载完成，实验设置: {exp_setting}")
            self.print_dataset_statistics(train, query, gallery)
            
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    
    def _read_id_file(self, id_file_path):
        with open(id_file_path, 'r') as f:
            content = f.read().strip()
            
        if ',' in content:  # 逗号分隔的ID列表
            ids = [int(pid_str) for pid_str in content.split(',')]
        else:  # 一行一个ID
            ids = [int(pid_str) for pid_str in content.splitlines() if pid_str.strip()]
            
        # 格式化ID为4位数字字符串，与文件夹命名一致
        return ["%04d" % pid for pid in ids]
        
    def _collect_imgs(self, pid_list, camera_list, relabel=False, pid2label=None):
        """收集指定ID和相机的所有图像"""
        dataset = []
        pid_container = set()
        
        # 如果需要重新标记ID但没有提供pid2label映射
        if relabel and pid2label is None:
            for pid_str in pid_list:
                pid_container.add(int(pid_str))
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        
        for pid_str in sorted(pid_list):
            pid_int = int(pid_str)
            
            for cam_name in camera_list:
                camid = self.cam_name_to_id[cam_name]
                img_dir = osp.join(self.dataset_dir, cam_name, pid_str)
                
                if not osp.isdir(img_dir):
                    continue
                    
                img_names = sorted(os.listdir(img_dir))
                for img_name in img_names:
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                        
                    img_path = osp.join(img_dir, img_name)
                    if relabel:
                        pid_label = pid2label[pid_int]
                    else:
                        pid_label = pid_int
                        
                    dataset.append((img_path, self.pid_begin + pid_label, camid, 0))  # track_id=0
                    
        return dataset
        
    def _process_train(self, relabel=True):
        """处理训练数据，合并train_id和val_id"""
        train_ids = self._read_id_file(self.train_id_file)
        val_ids = self._read_id_file(self.val_id_file)
        
        # 合并训练和验证ID
        all_train_ids = sorted(list(set(train_ids + val_ids)))
        
        # 收集所有相机的训练图像
        return self._collect_imgs(all_train_ids, self.all_cameras, relabel=relabel)
        
    def _process_query_rgb(self):
        """处理RGB查询集"""
        test_ids = self._read_id_file(self.test_id_file)
        return self._collect_imgs(test_ids, self.rgb_cameras, relabel=False)
        
    def _process_gallery_ir(self):
        """处理IR图库集"""
        test_ids = self._read_id_file(self.test_id_file)
        return self._collect_imgs(test_ids, self.ir_cameras, relabel=False)
        
    def _process_query_ir(self):
        """处理IR查询集"""
        test_ids = self._read_id_file(self.test_id_file)
        return self._collect_imgs(test_ids, self.ir_cameras, relabel=False)
        
    def _process_gallery_rgb(self):
        """处理RGB图库集"""
        test_ids = self._read_id_file(self.test_id_file)
        return self._collect_imgs(test_ids, self.rgb_cameras, relabel=False)