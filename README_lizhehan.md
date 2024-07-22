# 训练相关基本操作
## 环境
```
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple

```
## 数据集
+ 数据格式与yolov5相同
```
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val

```
## 训练
### 训练配置设置
+ configs文件夹下面有各种模型配置的文件，直接拷贝后修改即可
+ 可修改的配置有：预训练权重、模型高宽、骨干网络、颈部网络、头部网络、优化器、数据增强
### 命令行调用
#### 训练必用tools/train.py
+ --data-path 数据集加载，默认data文件夹下找yaml文件
+ --conf-file configs文件夹下找配置py文件
+ --img-size 训练尺度
+ --batch-size 视显存大小而定
+ --epochs 训练代数
+ --workers 0肯定可以，其他数值请自行尝试
+ --device gpu数量
+ --fuse_ab 设置训练时ab两条路，yolov6特有

#### 验证必用tools/eval.py
+ --data 数据集加载，默认data文件夹下找yaml文件
+ --weights 想要验证的权重文件地址，支持pt、torchscript、onnx、openvino、trt、tf、paddle
+ --batch-size 视验证目标而定
+ --conf-thres 置信度阈值，影响较大
+ --iou-thres iou阈值
+ --max-det 最大检测数量
+ --task 主要用val、test、speed
+ --device gpu数量
+ --workers 0肯定可以，其他数值请自行尝试
+ --half fp16推理

#### 推理必用tools/infer.py
+ --weights 想要验证的权重文件地址
+ --source 想要推理的目录，可以是图片、视频、文件夹、屏幕、摄像头
+ --img-size 推理尺度
+ --conf-thres 置信度阈值，影响较大
+ --iou-thres iou阈值
+ --max-det 最大检测数量
+ --device gpu数量
+ --not-save-img 不保存

#### 推理必用deploy/ONNX/export_onnx.py
+ --weights 想要导出的权重文件地址
+ --img-size 导出尺度
+ --batch-size 导出batch
+ --half 半精度
+ --simplify 调用onnx-simplify
+ --dynamic-batch 动态导出
+ --trt-version trt版本
+ --iou-thres iou阈值
+ --conf-thres 置信度阈值，影响较大
+ --device gpu数量

# 代码基础介绍
## configs模型配置文件
### base 基础模型
+ 仅常规卷积+relu
### experiment 实验模型
### mbla 使用MBLA主干
### qarepvgg repvgg的改进版
### repopt 重参数化结构量化优化
### yolov6_lite 轻量版本
### .py
+ 有finetune为迁移学习微调
+ 无finetune为从零训练
## data数据相关
+ .yaml和yolov5一致
## deploy部署
### NCNN
+ NCNN的导出torchscript和推理代码，详细内容参见文件夹下REDAME
### ONNX
+ 提供onnx导出、e2e导出和trt导出，详细内容参见文件夹下REDAME
+ trt推理
### openvino
+ 提供openvino导出，详细内容参见文件夹下REDAME
### tensorrt
+ 详细内容参见文件夹下REDAME
+ calibrator.py校准方法
+ eval_yolo_trt.py推理trt
+ onnx_to_trt.py从onnx转trt
+ processor.py前处理
+ tensorrt_processor.py trt的前处理
+ visulaize.py 可视化
+ yolov6.cpp 推理代码
## docs说明文件
## tools工具
### partial_quantization 部分量化
+ 部分量化，解决ptq量化大幅掉点问题，参见readme
+ 从掉7个点降低到掉0.3个点
### qat 训练后量化
+ 从掉2个点降低到掉0.1个点
### quantization 其他量化框架
+ mnn：暂缺
+ ppq：ppq量化接口
+ tensorrt：trt量化接口
### train.py训练
### eval.py验证
### infer.py推理
## yolov6核心代码
### assigners 各种anchor分配机制
### core 训练验证推理类
### data 数据加载、数据增强、可视化工具
### layers 基础模块，类似yolov5的common
### models
#### heads检测头类
#### losses损失函数类
#### .py各类模型框架，类似yolov5的yolo
### solver
### utils工具

# 特点介绍
## 主干
+ 参考repvgg，设计efficientrep，在小模型用repblock，在大模型上用csp stackrep blocks
## 颈部
+ 使用repblock替换csp
## head
+ 解耦+轻量化
## anchor_free
## simota标签分配定义正负样本
## TAL任务对齐学习
## VFL分类损失、SIOU/GIOU回归损失
## 自蒸馏
## repopt重参优化，改善ptq效果
## 量化敏感分析
## 引入qat
