# 参考https://github.com/Chien-Hung/DetVisGUI
# 其中批量测试，每点一张图片，切换的时候都重新调用检测模型和检测过程，可能有点消耗资源
import argparse
import itertools

import os
import os.path as osp
import time

from tkinter import (END, Button, Checkbutton, E, Entry, IntVar, Label,
                     Listbox, Menu, N, S, Scrollbar, StringVar, Tk, W, ttk)

from tkinter import filedialog
from tkinter.filedialog import askdirectory

import cv2
import matplotlib
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import collect_env, get_root_logger

import numpy as np
import platform
import pycocotools.mask as maskUtils  # not necessary in fact
from PIL import Image, ImageTk

matplotlib.use('agg')

# 定义在外面的函数，在所有类中都可以调用
# 解析运行时参数
def parse_args():
    parser = argparse.ArgumentParser(description='Caltech_GUI')
    parser.add_argument('--config',
                        default='./config/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py',
                        help='config file path')
    parser.add_argument('--ckpt',
                        default='./checkpoints/pretrain_attention_epoch_12.pth',
                        help='checkpoint file path')
    parser.add_argument('--img_root',
                        default='.data/test',  # 如果没设置img_root，它会有一个缺省值
                        help='test image path')
    parser.add_argument('--device', default='cuda', help='inference device')  # look like not necessary

    parser.add_argument('--no_gt',
                        default=True,
                        help='test_ images without groundtruth')
    parser.add_argument(
        '--det_box_color', default=(255, 255, 0), help='detection box color'
    )
    parser.add_argument(
        '--gt_box_color',
        default=(255, 255, 255)
    )
    parser.add_argument('--output', default='output', help='image save folder')  # 在我设计的程序里，必须设置output(默认保存路径)

    args = parser.parse_args()
    return args


# 把获得的检测结果转化成可读形式
def get_dets(det_results):
    det_results = np.asarray(det_results, dtype=object)
    return det_results


# 所有的检测框
def draw_all_det_boxes(img, single_detection, aug_category, threshold, img_width, img_height):
    args = parse_args()

    for idx, cls_objs in enumerate(single_detection):
        # category = data_info.aug_category.category[idx]   # notice
        category = aug_category.category[idx]

        for obj_idx, obj in enumerate(cls_objs):
            [score, box] = [round(obj[4], 2), obj[:4]]  # 置信度保留两位小数

            if score >= threshold:
                box = list(map(int, list(map(round, box))))  # round函数默认返回整数
                xmin = max(box[0], 0)
                ymin = max(box[1], 0)
                xmax = min(box[2], img_width)
                ymax = min(box[3], img_height)  # notice

                font = cv2.FONT_HERSHEY_SIMPLEX
                text = category + ':' + str(score)

                if ymax + 30 >= img_height:
                    cv2.rectangle(
                        img, (xmin, ymin),
                        (xmin + len(text) * 9, int(ymin - 20)),
                        (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, text, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1
                                )
                else:
                    cv2.rectangle(
                        img, (xmin, ymax),
                        (xmin + len(text) * 9, int(ymax + 20)),
                        (0, 0, 255), cv2.FILLED
                    )
                    cv2.putText(img, text, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), args.det_box_color, 2)

    return img


# 清理原来的并把新的|检测框信息|展示到列表上
def clear_add_listBox_obj(listBox_obj, dets, aug_category, threshold, listBox_obj_info):
    listBox_obj.delete(0, 'end')

    single_detection = dets

    num = 0
    for idx, cls_objs in enumerate(single_detection):
        category = aug_category.category[idx]

        for obj_idx, obj in enumerate(cls_objs):
            score = np.round(obj[4], 2)
            if score >= threshold:
                listBox_obj.insert('end', category + ":" + str(score))
                num += 1

    listBox_obj_info.set('检测到的目标数量 : {:3}'.format(num))
    # return num


# 用于读取数据集
class COCO_dataset:

    # def __init__(self,cfg,args):
    def __init__(self, args, img_root):
        # self.dataset = 'COCO'
        self.dataset = 'Caltech'  # just to display
        # self.img_root = getattr(cfg.data,args.stage).img_prefix
        # self.anno_root = getattr(cfg.data,args.stage).annfile  # 通过配置文件取真实标注，stage可在前面添加参数设置train或val或test数据集
        # self.img_root = args.img_root  # 注意这里直接从运行参数获得img_root而不是间接通过配置文件，若需要后续选择，那么应在后续赋值
        self.img_root = img_root

        self.config_file = args.config  # not necessary
        self.checkpoint_file = args.ckpt  # notice here
        self.mask = False  # not necessary
        self.device = args.device  # not necessary

        # according json to get category,image list,and annotations.
        self.img_list = self.get_img_list()  # notice

        self.aug_category = aug_category(['person', ])  # 注意到这里，未实例化就直接使用了aug_category

    def get_img_list(self):
        img_list = list()
        for image in sorted(os.listdir(self.img_root)):
            img_list.append(image)

        return img_list

    def get_img_by_name(self, name):
        img = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        return img

    # not necessary
    def get_img_by_index(self, idx):
        img = Image.open(os.path.join(self.img_root, self.img_list[idx])).convert('RGB')

        return img


# 用于定义字典中的类别名称
class aug_category:

    def __init__(self, categories):
        self.category = categories
        self.combo_list = categories.copy()  # not necessary
        self.combo_list.insert(0, 'All')  # not necessary
        self.all = True  # not necessary

window_title=''
config=''
ckpt=''
# 初始界面，选择单张测试或批量测试
class initface():

    def __init__(self):
        # def __init__(self,window):

        self.window = Tk()
        self.menubar = Menu(self.window)

        self.threshold = np.float32(0.5)  # 好像不需要
        self.tell_label1 = Label(
            self.window,
            font=("Times New Roman", 11),
            bg='white',
            width=30,
            height=1,
            text='请选择系统工作区')
        self.tell_label1.place(
            x=75,
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
            self.window, text='确 定', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.to_test)

        self.confirm_button.place(
            x=180,
            y=180
        )

    def select_work_space(self):
        self.work_space = askdirectory()
        self.work_space_label['text'] = self.work_space

    def to_test(self):
        self.tell_label1.destroy()
        self.work_space_label.destroy()
        self.work_space_button.destroy()
        self.confirm_button.destroy()
        global config
        config = self.work_space + '/config/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py'
        global ckpt
        ckpt = self.work_space + '/checkpoints/pretrain_attention_epoch_12.pth'
        global window_title
        window_title = '基于注意力机制的的行人目标检测系统'
        self.window.title(window_title)

        self.tell_label3 = Label(
            self.window,
            font=('Arial', 11),
            bg='white',
            width=30,
            height=1,
            text='请选择单张测试或批量测试'
        )
        self.tell_label3.place(
            x=60,
            y=60
        )

        self.one_test_button = Button(
            self.window, text='单张测试', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.to_one_test)
        self.one_test_button.place(x=80, y=140)

        self.many_test_button = Button(
            self.window, text='批量测试', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.to_many_test)
        self.many_test_button.place(x=240, y=140)

    def run(self):
        self.window.title('基于注意力机制的的行人目标检测系统')
        self.window.geometry('420x250')
        self.window.configure(background='white')
        self.window.mainloop()

    def to_one_test(self):
        self.window.destroy()
        window1 = Tk()
        one_test(window1).run()

    def to_many_test(self):
        self.window.destroy()
        window2 = Tk()
        many_test(window2).run()


# 单张测试
class one_test():

    def __init__(self, window1):
        self.window = window1

        self.select_pic_button = Button(
            self.window, text='选择图片', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.select_pic
        )
        self.pic_label = Label(
            self.window,
            font=('Arial', 11),
            bg='LightGrey',
            width=65,
            height=1
        )
        self.th_label = Label(
            self.window,
            font=('LightSkyBlue', 11),
            bg='LightSkyBlue',
            width=10,
            height=1,
            text='置信度阈值：'
        )
        self.th_label_tip = Label(
            self.window,
            font=('Arial', 7),
            bg='white',
            fg='gray',
            text='范围应在0-1之间')
        self.th_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10
        )
        self.start_test_button = Button(
            self.window, text='开始检测', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.test
        )

        self.test_result_label = Label(
            self.window,
            bg='white',
            width=10,
            height=1,
            text='检测结果：',
            font=('Arial', 11)
        )
        self.listBox_obj_info = StringVar()  # notice

        self.listBox_obj_label1 = Label(
            self.window,
            font=('Arial', 11),
            bg='LightSkyBlue',
            width=39,
            height=1,
            textvariable=self.listBox_obj_info
        )
        self.listBox_obj_label2 = Label(
            self.window,
            font=('Arial', 11),
            bg='LightSkyBlue',
            width=39,
            height=1,
            text='数据格式(目标类别：置信度)'
        )
        self.save_one_button = Button(self.window, text='保存检测结果', height=1, font=('Arial', 11),
                                      bg='LightSkyBlue', command=self.save_one)
        # 调用clear_add_listBox_obj的时候会把具体的|缺陷种类和置信度|写进列表
        self.listBox_obj = Listbox(
            self.window, bg='white', width=50, height=12, font=('Times New Roman', 10)
        )

        self.label_img = Label(self.window, bg='white')

    # 选择图片
    def select_pic(self):
        self.pic = filedialog.askopenfilename()
        self.pic_label['text'] = self.pic

    # 推理检测
    def test(self):
        self.threshold = float(self.th_entry.get())
        # 这里是通过entry.get()的方式获取和改变threshold，而不是通过change_threshold。不过change_threshold_button和th_button有什么不同呢，噢change_threshold_button是鼠标上下键控制触发的
        self.args = parse_args()

        self.aug_category = aug_category(['person',])  # I add this

        self.img = Image.open(self.pic).convert('RGB')  # self.pic是路径
        self.img_width, self.img_height = self.img.width, self.img.height
        img = np.asarray(self.img)

        self.model = init_detector(
            config,  # 这里能不能self.args.config.什么来引用配置文件里的某项参数、给它赋值？
            ckpt,
            device=self.args.device
        )

        result = inference_detector(self.model, img)
        self.dets = get_dets(result)
        self.img = draw_all_det_boxes(img, self.dets, self.aug_category, self.threshold, self.img_width, self.img_height)
        clear_add_listBox_obj(self.listBox_obj, self.dets, self.aug_category, self.threshold, self.listBox_obj_info)
        self.show_img = self.img  # 如果需要后面可以保存
        img = Image.fromarray(self.img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)

    def save_one(self):

        self.output = self.args.output
        # self.img_name = os.path.basename(self.pic).split('.')[0]
        self.img_name = os.path.basename(self.pic)
        cv2.imwrite(
            os.path.join(self.output, self.img_name),
            cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB)
        )

    def run(self):
        self.window.title(window_title)
        self.window.geometry('750x450')
        self.window.configure(background='white')
        self.select_pic_button.place(x=40, y=36)
        self.pic_label.place(x=140, y=40)
        self.th_label.place(x=40, y=101)
        self.th_label_tip.place(x=40, y=130)
        self.th_entry.place(x=140, y=100)
        self.start_test_button.place(x=240, y=96)
        self.test_result_label.place(x=450, y=100)
        self.listBox_obj_label1.place(x=40, y=150)
        self.listBox_obj_label2.place(x=40, y=180)
        self.listBox_obj.place(x=40, y=200)
        self.label_img.place(x=450, y=130)
        self.save_one_button.place(x=50, y=380)

# 批量测试
class many_test():
    def __init__(self, window2):
        self.window = window2
        self.args = parse_args()  # 留意配置的读取（数据路径）

        self.info = StringVar()
        self.info_label = Label(self.window, bg='LightSkyBlue', width=4,
                                textvariable=self.info)  # 好像没有用到，应该是点击图片列表的时候，标题为xx数据集：xxx.jpg
        self.listBox_img = Listbox(
            self.window, width=50, height=25, font=('Times New Roman', 10)
        )
        self.listBox_obj = Listbox(
            self.window, width=50, height=12, font=('Times New Roman', 10)
        )
        self.scrollbar_img = Scrollbar(
            self.window, width=15, orient='vertical'
        )
        self.scrollbar_obj = Scrollbar(
            self.window, width=15, orient='vertical'
        )
        self.listBox_img_info = StringVar()
        self.listBox_img_label = Label(
            self.window,
            font=('Arial', 11),
            bg='LightSkyBlue',
            width=38,
            height=1,
            textvariable=self.listBox_img_info
        )

        self.listBox_obj_info = StringVar()
        self.listBox_obj_label1 = Label(
            self.window,
            font=('Arial', 11),
            bg='LightSkyBlue',
            width=39,
            height=1,
            textvariable=self.listBox_obj_info
        )
        self.listBox_obj_label2 = Label(
            self.window,
            font=('Arial', 11),
            bg='LightSkyBlue',
            width=39,
            height=1,
            text='数据格式(目标类别：置信度)'
        )

        self.aug_category = aug_category(['person', ])

        self.show_det_txt = IntVar(value=1)
        self.checkbn_det_txt = Checkbutton(
            self.window,
            text='文字',
            font=('Arial',10,'bold'),
            variable=self.show_det_txt,
            command=self.change_img,
            fg='LightSkyBlue'
        )
        self.show_dets = IntVar(value=1)
        self.checkbn_det = Checkbutton(
            self.window,
            text='检测',
            font=('Arial',10,'bold'),
            variable=self.show_dets,
            command=self.change_img,
            fg='LightSkyBlue'
        )

        self.dataPath_button = Button(
            self.window, text='选择测试数据路径', font=('Arial', 11), bg='LightSkyBlue', height=1, command=self.select_dataPath
        )
        self.dataPath_label = Label(
            self.window,
            font=('Arial', 11),
            bg='LightGrey',
            width=38,
            height=1
        )

        self.th_label = Label(self.window,
                              font=('Arial', 11),
                              bg='LightSkyBlue',
                              width=18,
                              height=1,
                              text='置信度阈值'
                              )
        self.th_label_tip = Label(
            self.window,
            font=('Arial', 7),
            bg='white',
            fg='gray',
            text='范围应在0-1之间')
        self.threshold = np.float32(0.5)

        self.th_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10,
            textvariable=StringVar(self.window, value=str(self.threshold))
        )
        self.th_button = Button(
            self.window, text='Enter', height=1, command=self.change_threshold
        )

        self.find_label = Label(
            self.window,
            font=('Arial', 11),
            bg='LightSkyBlue',
            width=13,
            height=1,
            text='搜索'
        )

        self.find_name = ''
        self.find_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=15,
            textvariable=StringVar(self.window, value=str(self.find_name))
        )
        self.find_button = Button(
            self.window, text='Enter', height=1, command=self.findname
        )

        self.save_all_button = Button(self.window, text='保存所有图片', height=1, font=('Arial', 11),
                                      bg='LightSkyBlue', command=self.save_all)

        self.listBox_img_idx = 0  # notice

        self.img_name = ''  # 后面用于判断是否属于文件夹列表中、保存图片等
        self.show_img = None  # 好像没用到
        self.output = self.args.output  # 图片保存的路径，判断若图片已保存则在列表中显示为指定背景色
        self.model = init_detector(
               config, ckpt, device=self.args.device
        )

        if not os.path.isdir(self.output):
            os.makedirs(self.output)

        self.button_clicked = False
        self.label_img = Label(self.window, bg='white')  # notice,I add this，定义label的时候可以先不赋值text或image吗？

    # 选择测试集路径
    def select_dataPath(self):
        self.dataPath = filedialog.askdirectory()
        self.dataPath_label['text'] = self.dataPath
        self.data_info = COCO_dataset(self.args, self.dataPath)
        self.img_list = self.data_info.img_list
        self.clear_add_listBox_img()  # I add this,如果要在这里调用clear_add_listBox_img，那么应提前定义self.label_img(也是为了避免每次调用change_img都创建一个新的实例，太多实例耗内存）

        # 或者就学原程序，这里先定义self.label_img、只显示img_list的第一张图片??；好像也没用，得在定义函数之前、类的初始化中定义self.label_img才能做到只创建一个Label实例，但是类初始化的时候还无法获得img_list

        # detect and save all
        # 原程序先clear_add_listBox_img ，只有先设定鼠标选择某张图片，再调用change_img的时候才检测和画图，而我这里需要在首次获取测试数据路径时?全部检测画图保存；要不，添加一个保存所有图片的按钮
        '''
        for items in self.img_list:


            # name = self.listBox_img.get(self.listBox_img_idx)

            # img = self.data_info.get_img_by_name(name)  # 还需要img_root的引入才能调用这个

            # self.img_name = name
            self.img_name = os.path.basename(items)

            img = items  # I add this
            self.img = img

            result = inference_detector(self.model, img)
            self.dets = get_dets(result)

            # if self.show_dets.get():
            img = draw_all_det_boxes(img, self.dets, self.data_info, self.threshold, self.img_width,
                                         self.img_height)  # 注意到这里的传参

            self.show_img = img  # 好像不需用到


            cv2.imwrite(
                os.path.join(self.output, self.img_name),
                cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB)
            )
            '''

    def save_all(self):
        self.output = self.args.output
        # self.img_name = os.path.basename(self.pic).split('.')[0]

        for items in sorted(os.listdir(self.dataPath)):
            self.img_name = os.path.basename(items)

            self.img = os.path.join(self.dataPath, self.img_name)
            self.img = Image.open(self.img).convert('RGB')  # self.img是路径
            self.img_width, self.img_height = self.img.width, self.img.height

            img = np.asarray(self.img)

            result = inference_detector(self.model, img)
            self.dets = get_dets(result)

            # if self.show_dets.get():
            img = draw_all_det_boxes(img, self.dets, self.aug_category, self.threshold, self.img_width,
                                     self.img_height)  # 注意到这里的传参

            self.show_img = img

            cv2.imwrite(
                os.path.join(self.output, self.img_name),
                cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB)
            )

    # 改变置信度阈值
    def change_threshold(self, event=None):
        try:
            self.threshold = np.float32(self.th_entry.get())  # 在draw画缺陷框的时候就会调用这个self.threshold来判断过滤
            self.change_img()

            if self.window.focus_get() == self.listBox_obj:
                self.listBox_obj.focus()
            else:
                self.listBox_img.focus()

            self.button_clicked = True  # notice
        except ValueError:
            self.window.title('Please enter a number as score threshold.')

    # 根据鼠标点击的来检测列表中的相应的图片，画出并显示检测框信息
    def change_img(self, event=None):
        if len(self.listBox_img.curselection()) != 0:
            self.listBox_img_idx = self.listBox_img.curselection()[0]

        self.listBox_img_info.set('图片{:6}/{:6}'.format(self.listBox_img_idx + 1, self.listBox_img.size()))

        print(self.listBox_img_idx)
        name = self.listBox_img.get(self.listBox_img_idx)
        print(name)

        img = self.data_info.get_img_by_name(name)  # 还需要img_root的引入才能调用这个
        self.img_width, self.img_height = img.width, img.height

        img = np.asarray(img)
        print(img)

        self.img_name = name
        # self.img = img

        result = inference_detector(self.model, img)
        self.dets = get_dets(result)

        img = draw_all_det_boxes(img, self.dets, self.aug_category, self.threshold, self.img_width, self.img_height)
        clear_add_listBox_obj(self.listBox_obj, self.dets, self.aug_category, self.threshold, self.listBox_obj_info)

        self.show_img = img  # 好像不需用到
        img = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(img)

        self.label_img.config(image=self.photo)

        self.window.update_idletasks()  # notice

        # 如果自动保存每张图片，那么这一段不要了
        if self.img_name in os.listdir(self.output):
            self.listBox_img_label.config(bg='#CCFF99')
        else:
            self.listBox_img_label.config(bg='#87CEFA')

    # draw_one_det_boxes函数其实是供change_obj函数调用（当鼠标点击检测框列表中的某个检测框的信息时）
    def draw_one_det_boxes(self, img, single_detection, selected_idx=-1):
        idx_counter = 0
        for idx, cls_objs in enumerate(single_detection):
            category = self.aug_category.category[idx]
            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    # notice here
                    if idx_counter == selected_idx:
                        box = list(map(int, list(map(round, box))))
                        xmin = max(box[0], 0)
                        ymin = max(box[1], 0)
                        xmax = min(box[2], self.img_width)
                        ymax = min(box[3], self.img_height)

                        if self.show_det_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + ':' + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(
                                    img, (xmin, ymin),
                                    (xmin + len(text) * 9, int(ymin - 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1
                                            )
                            else:
                                cv2.rectangle(
                                    img, (xmin, ymax),
                                    (xmin + len(text) * 9, int(ymax + 20)),
                                    (0, 0, 255), cv2.FILLED
                                )
                                cv2.putText(img, text, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.args.det_box_color, 2)

                        return img

                    else:
                        idx_counter += 1  # notice here

    # 当鼠标点击检测框列表中的某个检测框的信息时
    def change_obj(self, event=None):
        if len(self.listBox_obj.curselection()) == 0:
            self.listBox_img.focus()
            return
        else:
            listBox_obj_idx = self.listBox_obj.curselection()[0]

        # 把检测到的信息以列表方式呈现
        self.listBox_obj_info.set('检测到的目标：{:4}/{:4}'.format(
            listBox_obj_idx + 1, self.listBox_obj.size()
        ))
        name = self.listBox_img.get(self.listBox_img_idx)  # 留意这个self.listBox_img_idx，可能没有self
        img = self.data_info.get_img_by_name(name)  # 这个也要留意
        self.img_width, self.img_height = img.width, img.height
        img = np.asarray(img)
        self.img_name = name
        self.img = img

        # 把图片画出来
        if self.show_dets.get():
            img = self.draw_one_det_boxes(img, self.dets, listBox_obj_idx)

        self.show_img = img
        img = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)  # 留意这里，看来self.label_img得提前定义？是可以提前定义的
        self.window.update_idletasks()  # notice

        if self.img_name in os.listdir(self.output):
            self.listBox_img_label.config(bg='#CCFF99')
        else:
            self.listBox_img_label.config(bg='LightSkyBlue')

    # 缩放图片
    def scale_img(self, img):
        [s_w, s_h] = [1, 1]

        [fix_width, fix_height] = [220, 220]

        # change the image size according to window size
        if self.window.winfo_width() != 1:
            fix_width = (
                    self.window.winfo_width() - self.listBox_img.winfo_width() -
                    self.scrollbar_img.winfo_width() - 5
            )
            fix_height = 750

        # handle image size that is too big
        if img.width > fix_width:
            s_w = fix_width / img.width
        if img.height > fix_height:
            s_h = fix_height / img.height

        scale = min(s_w, s_h)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.ANTIALIAS)

        return img

    # 键盘左右键响应
    def change_threshold_button(self, v):
        self.threshold += v
        if self.threshold <= 0:
            self.threshold = 0
        elif self.threshold >= 1:
            self.threshold = 1

        self.th_entry.delete(0, END)
        self.th_entry.insert(0, str(round(self.threshold, 2)))
        self.change_threshold()

    # 保存图片
    def save_img(self):
        print('Save image to' + os.path.join(self.output, self.img_name))
        cv2.imwrite(
            os.path.join(self.output, self.img_name),
            cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB)
        )
        self.listBox_img_label.config(bg='#CCFF99')

    # 键盘事件响应
    def eventhandler(self, event):
        entry_list = [self.find_entry, self.th_entry]
        if self.window.focus_get() not in entry_list:
            if platform.system() == 'Windows':
                state_1key = 8
                state_2key = 12
            else:
                state_1key = 16
                state_2key = 20

            if event.state == state_1key and event.keysym == 'left':
                self.change_threshold_button(-0.1)
            elif event.state == state_1key and event.keysym == 'right':
                self.change_threshold_button(0.1)
            elif event.keysym == 'q':
                self.window.quit()
            elif event.keysym == 's':
                self.save_img()

            # 没看懂以下这段
            if self.button_clicked:
                self.button_clicked = False
            else:
                if event.keysym in ['KP_Enter', 'Return']:
                    self.listBox_obj.focus()
                    self.listBox_obj.select_set(0)
                elif event.keysym == 'Escape':
                    self.change_img()
                    self.listBox_img.focus()

    # 改变显示的缺陷类别
    def combobox_change(self, event=None):
        self.listBox_img.focus()
        self.change_img()

    # 重新载入测试集图片列表
    def clear_add_listBox_img(self):
        self.listBox_img.delete(0, 'end')  # delete listBox_img 0 ~ end items

        for item in self.img_list:
            # for item in self.data_info.img_list:
            self.listBox_img.insert('end', item)  # notice here,get the img name

        self.listBox_img.select_set(0)  # notice
        self.listBox_img.focus()  # 好像不是必须的
        self.change_img()  # detect through this function

    # 以图片名搜索图片
    def findname(self, event=None):
        self.find_name = self.find_entry.get()
        new_list = []

        if self.find_name == '':
            new_list = self.data_info.img_list
        else:
            for img_name in self.data_info.img_list:
                if self.find_name[0] == '!':
                    if self.find_name[1:] not in img_name:
                        new_list.append(img_name)  # 选择不含某字段的图片名
                else:
                    if self.find_name in img_name:
                        new_list.append(img_name)
        if len(new_list) != 0:
            self.img_list = new_list
            self.clear_add_listBox_img()
            # self.clear_add_listBox_obj()
            clear_add_listBox_obj(self.listBox_obj, self.dets, self.aug_category, self.threshold, self.listBox_obj_info)
            self.button_clicked = True
        else:
            self.window.title("Can't find any image about '{}'".format(self.find_name))

    def run(self):
        self.window.title(window_title)
        self.window.geometry('800x550')
        self.window.configure(background='white')
        # self.window.mainloop()

        self.scrollbar_img.config(command=self.listBox_img.yview())

        self.listBox_img.config(yscrollcommand=self.scrollbar_img.set)

        self.scrollbar_obj.config(command=self.listBox_obj.yview())
        # self.scrollbar_obj.set(0.5,1)

        self.listBox_obj.config(yscrollcommand=self.scrollbar_obj.set)
        self.dataPath_button.place(x=10, y=5)
        self.dataPath_label.place(x=10, y=36)
        self.listBox_img_label.place(x=10, y=58)
        self.find_label.place(x=10, y=80)
        self.find_entry.place(x=143, y=80)
        self.find_button.place(x=295, y=80)
        self.scrollbar_img.place(x=342, y=110)  # notice,this one
        self.listBox_img.place(x=5, y=110)

        # self.label_img.place(x=400,y=260,anchor=N+W)

        self.th_label.place(x=400, y=5)
        self.th_label_tip.place(x=400, y=25)
        self.th_entry.place(x=580, y=5)
        self.th_button.place(x=690, y=5)
        self.listBox_obj_label1.place(x=400, y=40)  # 根据缺陷数量变化
        self.listBox_obj_label2.place(x=400, y=65)  #
        self.scrollbar_obj.place(x=736, y=85)
        self.listBox_obj.place(x=400, y=85)
        self.label_img.place(x=400, y=265)
        self.save_all_button.place(x=50, y=500)

        # self.clear_add_listBox_img()

        # 给一些部件绑定鼠标响应事件
        self.listBox_img.bind('<<ListboxSelect>>', self.change_img)
        # self.listBox_img.bind_all('<KeyRelease>',self.eventhandler)  # can save the picture

        self.listBox_obj.bind('<<ListboxSelect>>', self.change_obj)

        self.th_entry.bind('<Return>', self.change_threshold)
        self.th_entry.bind('<KP_Enter>', self.change_threshold)

        self.find_entry.bind('<Return>', self.findname)
        self.find_entry.bind('<KP_Enter>', self.findname)



if __name__ == '__main__':
    initface().run()
    # window = Tk()
    # initface(window).run()