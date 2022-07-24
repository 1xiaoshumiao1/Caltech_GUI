# __author__ = 'ChienHung Chen in Academia Sinica IIS'

import argparse
import itertools
import json
import os
import os.path as osp
import subprocess
import torch
import time

import pickle
import xml.etree.ElementTree as ET
from tkinter import (END, Button, Checkbutton, E, Entry, IntVar, Label,
                     Listbox, Menu, N, S, Scrollbar, StringVar, Tk, W, ttk)
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askdirectory

import cv2
import matplotlib
import mmcv
from mmdet.apis import init_detector, inference_detector  # mmdet也是一个已经安装上的包
from mmdet.utils import collect_env, get_root_logger

from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

import numpy as np
import platform
import pycocotools.mask as maskUtils
from PIL import Image, ImageTk

matplotlib.use('TkAgg')


"""
def parse_args():
    parser = argparse.ArgumentParser(description='defectGUI')
    parser.add_argument('--config',
                        default='/home/ubuntu/detectGUI/config/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py',
                        # this in runtime ops
                        help='config file path')

    parser.add_argument('--ckpt',
                        default='./checkpoints/pretrain_attention_ms600_epoch_12.pth',  # this in runtime ops
                        help='checkpoint file path')

    parser.add_argument('--img_root',  # this in runtime ops
                        default='./data/valAndTest',
                        help='test image path')

    parser.add_argument('--device', default='cuda', help='inference device')

    parser.add_argument(
        '--no_gt',
        default=True,
        help='test images without groundtruth')

    parser.add_argument(
        '--det_box_color', default=(255, 255, 0), help='detection box color')

    parser.add_argument(
        '--gt_box_color',
        default=(255, 255, 255),
        help='groundtruth box color')

    parser.add_argument('--output', default='output', help='image save folder')

    args = parser.parse_args()
    return args
"""

class COCO_dataset:

    def __init__(self, config, args):
        # self.dataset = 'COCO'
        self.dataset = 'Celtech'
        self.config_file = config
        self.mask = False
        self.device = args.device
        self.aug_category = aug_category([
              'person',
        ])

window_title=''

class initface:
    def __init__(self, master):

        self.window = master

        self.tell_label1 = Label(
            self.window,
            font=("Times New Roman", 11),
            bg='white',
            width=30,
            height=1,
            text='请选择系统工作区')
        self.tell_label1.place(
            x=85,
            y=60
        )

        self.work_space_label = Label(
            self.window,
            font=('Arial ', 11),
            bg='LightGrey',
            width=32,
            height=1,
        )
        self.work_space_label.place(
            x=40,
            y=120
        )

        self.work_space_button = Button(
            self.window, text='选 择', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.select_work_space)

        self.work_space_button.place(
            x=340,
            y=115
        )

        self.confirm_button = Button(
             self.window, text='确 定', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.to_train)

        self.confirm_button.place(
            x=180,
            y=180
        )

    def select_work_space(self):
        self.work_space = askdirectory()
        self.work_space_label['text'] = self.work_space

    def to_train(self):
        self.tell_label1.destroy()
        self.work_space_label.destroy()
        self.work_space_button.destroy()
        self.confirm_button.destroy()

        self.config=self.work_space+'/config/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py'
        global window_title
        window_title='基于注意力机制的的Faster R-cnn训练'

        self.window.title(window_title)

        self.window.geometry('500x250')

        self.model_save_label1 = Label(
            self.window,
            font=('Arial', 11),
            bg='white',
            width=13,
            height=1,
            text='训练结果保存路径')
        self.model_save_label1.place(
            x=40,
            y=20
        )

        self.model_save_button = Button(
            self.window, text='选 择', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.select_model_save)
        self.model_save_button.place(
            x=400,
            y=48
        )

        self.model_save_label2 = Label(
            self.window,
            font=('Arial', 11),
            bg='LightGrey',
            width=38,
            height=1
        )
        self.model_save_label2.place(
            x=40,
            y=50
        )

        self.epoch_label = Label(
            self.window,
            font=('Arial', 11),
            bg='white',
            width=7,
            height=1,
            text='迭代次数')
        self.epoch_label.place(
            x=40,
            y=100
        )
        self.epoch_label_tip = Label(
            self.window,
            font=('Arial', 8),
            bg='white',
            fg='gray',
            text='默认12,建议8-14')
        self.epoch_label_tip.place(
            x=40,
            y=125
        )
        self.epoch_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=5,
        )
        self.epoch_entry.place(
            x=120,
            y=100
        )

        self.lr_label = Label(
            self.window,
            font=('Arial', 11),
            bg='white',
            width=5,
            height=1,
            text='学习率')
        self.lr_label.place(
            x=240,
            y=100
        )
        self.lr_label_tip = Label(
            self.window,
            font=('Arial', 8),
            bg='white',
            fg='gray',
            text='默认0.02,建议0.0025*GPU数')
        self.lr_label_tip.place(
            x=240,
            y=125
        )
        self.lr_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=5,
        )
        self.lr_entry.place(
            x=300,
            y=100
        )
        self.batch_label = Label(
            self.window,
            font=('Arial', 11),
            bg='white',
            width=8,
            height=1,
            text='batch大小')
        self.batch_label.place(
            x=40,
            y=150
        )
        self.batch_label_tip = Label(
            self.window,
            font=('Arial', 8),
            bg='white',
            fg='gray',
            text='建议2,4,8等')
        self.batch_label_tip.place(
            x=40,
            y=175
        )
        self.batch_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=5,
        )
        self.batch_entry.place(
            x=120,
            y=150
        )
        self.start_train_button = Button(
            self.window, text='开 始 训 练', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.just_train)

        self.start_train_button.place(
            x=300,
            y=200
        )

    def select_model_save(self):

        self.model_save = askdirectory()
        self.model_save_label2['text'] = self.model_save

    def just_train(self):

        """
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        """
        cfg = mmcv.Config.fromfile(self.config)  # 这里需要导入另外的(maybe与test不同的)配置文件
        cfg.work_dir = self.model_save
        cpu_numbers= torch.cuda.device_count()
        if self.epoch_entry.get()!='':
            cfg.total_epochs=int(self.epoch_entry.get())
        if self.lr_entry.get() != '':
            cfg.optimizer.lr=float(self.lr_entry.get())
        if self.batch_entry.get() != '':
            cfg.data.samples_per_gpu=int(int(self.batch_entry.get())/cpu_numbers)
        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        datasets = [build_dataset(cfg.data.train)]
        model.CLASSES = datasets[0].CLASSES

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'

        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        train_detector(
            model,
            datasets,
            cfg,
            # distributed=distributed,
            distributed=False,
            # validate=(not args.no_validate),
            validate=True,
            timestamp=timestamp,
            meta=meta)
class basedesk:
    def __init__(self, master):
        self.window = master
        self.window.geometry('420x250')
        self.window.configure(background='white')
        self.window.title('Faster R-cnn模型训练系统')

        initface(self.window)


class aug_category:
    def __init__(self, categories):
        self.category = categories
        self.combo_list = categories.copy()
        self.combo_list.insert(0, 'All')
        self.all = True


if __name__ == '__main__':
    window = Tk()  #
    basedesk(window)
    window.mainloop()
    # vis_tool().run()


