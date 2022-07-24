代码安装好环境之后可以直接运行,在项目中,checkpoints里放的是训练完之后的权重文件,config里是mmdetection的模型配置文件,data里面是一个coco格式的行人数据集,如果训练自己的数据集需要进行替换,output是图片的默认保存路径,results作为训练结果的保存路径.本身是一个已经训练好的行人训练/检测系统.

person_train.py是训练交互代码,person_test.py是测试交互代码,具体使用可参照之前许老师提交的程序使用说明书.

server.py是一个检测系统服务端的代码也称为接口,client.py是一个调用接口的示例代码


#使用自己的数据集训练#
1、定义数据种类，需要修改的地方在mmdetection/mmdet/datasets/coco.py。把CLASSES的那个tuple改为自己数据集对应的种类tuple即可。例如：
    CLASSES = ('Glass_Insulator', 'Composite_Insulator', 'Clamp', 'Drainage_Plate')
    
2、接着在mmdetection/mmdet/core/evaluation/class_names.py修改coco_classes数据集类别，这个关系到后面test的时候结果图中显示的类别名称。例如：
	def coco_classes():
	    return [
		'Glass_Insulator', 'Composite_Insulator', 'Clamp', 'Drainage_Plate'
	    ]
	    
3、修改configs/faster_rcnn_r50_fpn_1x.py中的model字典中的num_classes、data字典中的img_scale,data中ann_file和imag_prefix路径。例如：
	num_classes=4,#类别数
	img_scale=(640,478), #输入图像尺寸的最大边与最小边（train、val、test这三处都要修改）


4.转到mmdetection目录下,在命令行重新编译修改才会生效,例如
    cd mmdetetion
    python setup.py develop
    
  
#使用交互界面
   需要将person_test中全局变量config和ckpt改为目前使用的文件路径, 将person_train.py中的self.config改为目前使用的路径
