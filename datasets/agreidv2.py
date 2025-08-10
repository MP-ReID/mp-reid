# encoding: utf-8
"""
@author:  huynguyen792
@contact: nguyet91@qut.edu.au
"""

import glob
import re
import mat4py
import pandas as pd
import torch

import os.path as osp

from .bases import BaseImageDataset


class AGReIDv2(BaseImageDataset):

    def __init__(self, root='nas_24/AG-ReID',
                 verbose=True, exp_setting=None, **kwargs):
        super(AGReIDv2, self).__init__()
        self.dataset_dir = root
        self.exp_setting = exp_setting

        if self.exp_setting is not None:
            self.split_file = osp.join(self.dataset_dir, f'{self.exp_setting}.txt')
        else:
            self.split_file = None

        self.train_dir = osp.join(self.dataset_dir, 'train_all')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.qut_attribute_path = osp.join(self.dataset_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")

        self._check_before_run()

        if self.split_file and osp.exists(self.split_file):
            query_list, gallery_list = self._process_split_file(self.split_file)
            train = self._process_dir(self.train_dir, is_train=True)
            query = self._process_img_list(query_list, is_train=False)
            gallery = self._process_img_list(gallery_list, is_train=False)

        if verbose:
            print("=> AG-ReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, *_ = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, *_ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, *_ = self.get_imagedata_info(self.gallery)

        # 兼容view_num
        self.num_train_vids = 1
        self.num_query_vids = 1
        self.num_gallery_vids = 1

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        data = []
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            # relabel pid
            pid = pid2label[pid]
            data.append((img_path, pid, camid, 0))
        return data

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3

        return dict_attribute

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False

    def _process_split_file(self, split_file):
        query, gallery = [], []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('query/'):
                    query.append(osp.join(self.dataset_dir, line))
                elif line.startswith('gallery/'):
                    gallery.append(osp.join(self.dataset_dir, line))
        return query, gallery

    def _process_img_list(self, img_list, is_train=True):
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')
        pid_container = set()
        for img_path in img_list:
            fname = osp.split(img_path)[-1]
            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        data = []
        for img_path in img_list:
            fname = osp.split(img_path)[-1]
            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            # relabel pid
            pid = pid2label[pid]
            data.append((img_path, pid, camid, 0))
        return data