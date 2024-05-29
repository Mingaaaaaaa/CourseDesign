# 基于 Opencv 的人脸表情识别系统

## 项目简介
为了满足学校课程设计的要求，所以写了这个程序
课程为南京邮电大学计算机科学与技术大三下的《图像与视频处理》
❤感谢 MaxtuneLee，基于他的课设，我使用 deepface 加入表情识别能力

课题的要求是与数字图像和视频处理相关，可涉及图像增强、形态学处理、图像分割、图像和视频压缩、质量评价等一个或多个授课内容，其他数字图像和视频处理领域内容也可。

## 程序架构
整体架构分为
- 视图层 View
  - 界面
  - 日志
- 状态层 Store
- 服务层 Service
  - 传统方法
  - MindSpore
  - 相机视频流读取与处理

## 如何看这个项目
由于我是前端开发，所以在qt开发上用了很多前端的思想，望传统的qt开发者见谅。
```python
> main.py
...
    store = ConfigStore() # 状态
    classicPreditor = ClassicFaceRecognizer() # 传统方法
    camera = VisionService(store) # 视觉服务：包含相机视频流读取与处理
    window = MainPage(camera.all_queues, classicPreditor, camera, store) # 视图层，界面逻辑这里看
...
```
从 main 函数进去看即可，需要更改哪个部分，或者添加方法可以直接在 VisionService 添加


祝有需要的同学高分

## 如何启动这个项目
1. pip install requirements.text
2. 需要下载https://github.com/serengil/tensorflow-101/blob/master/model/
facial_expression_model_weights.h5 后,将文件放到 C:\Users\{用户名}\.deepface\weights 目录下
3. 在dataset/full目录下创建学号Bxxxxxxxx 的文件夹，在文件夹内放入图片/启动程序录入
4. 训练
5. 打开摄像头
6. 打开人脸检测
7. 打开人脸识别，即开始
