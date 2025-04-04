# 通用卡证信息高精度识别流程 OCR 文本检测 文本识别 身份证 银行卡

没有足够的数据集绚训练的前提下，可以看看下面的思路和实现。

## 问题描述：
这里所说的“身份证件”是指具有头像照片和文字描述的卡片式证件，如：二代身份证、社保卡、医保卡、工牌等。	
在传统需要对身份证件进行扫描识别的场景，通常使用平板扫描或高拍仪等设备进行图像采集，以确保对扫描或拍照的光照、背景、文字方向等进行有效控制，从而有助于后续对身份证件上的文字和图片进行识别。
我们可以将此类场景定义为“封闭场景”的图像采集。在当前移动互联网时代，

许多需要进行图像采集的场景，例如使用各类手机App或微信公众号时，通常需要使用手机或平板设备的拍照功能进行采集，
因而对清晰度、环境光、背景、手持遮挡、摆放方向、倾斜角度等条件不能进行统一的控制。此类对外部条件不可控的图像采集场景，
我们定义为“开放场景”。

传统的机器学习方法可以完成大部分“封闭场景”的扫描识别任务，而对“开放场景”的扫描识别则不能达到实用的性能和结果。	

目前业务系统中最常见的“开放场景”应用是：用户通过手机对身份证件进行拍照上传，系统对上传的证件照片识别出身份信息，为用户自动填写相关的个人身份信息，
例如姓名、出生年月、身份证号等，从而免去了用户手工输入的麻烦。而在此类开放场景中拍摄的证件照片，通常因为上述不可控的因素，导致不能有效对身份证件进行定位、检测和识别。

因此，本方案针对“开放场景”提出了一个有效的解决方案，对输入的图片进行预处理，再利用paddleocr进行OCR识别。

## 思路：
图像预处理为了使得paddleocr输入的图片更容易被识别（仅留下身份证）。
那么思路可以是通过目标检测方法，将身份证的框框识别出来，进行裁剪。
或者可以是通过关键点矫正图像。
或者还可以是通过传统的图像预处理方法，增强图像，矫正裁剪。

#### 目标检测
[YOLO-World 模型](https://docs.ultralytics.com/zh/models/yolo-world/)
使用方法:
```
from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict("path/to/image.jpg")

# Show results
results[0].show()
```

#### 抠图算法
rembg 使用 MODNet 或类似的语义分割模型，已经在 大量带有透明通道的图像数据 上训练过。

主要用来 区分前景（主体）和背景。
#### paddleocr

https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/OCR.html#221

reference:

1.[开放场景的身份证件定位方法](https://mp.weixin.qq.com/s/z3JO6ujvqkVGSEB8ZohOsg?poc_token=HHU07mejovj2nunuVqlbJdoEi642XgTsxz3XwKOZ)

https://github.com/jack139/locard

2.[深度学习与图像处理 |身份证识读APP（关键点检测）](https://mp.weixin.qq.com/s/TceH_7nInlsMmiXn23YFbg)

3.openCV 图像处理

https://github.com/LeBron-Jian/ComputerVisionPractice?tab=readme-ov-file

