# 基于 Opencv 的人脸识别系统

## 项目简介
为了满足学校课程设计的要求，所以写了这个程序

课程为南京邮电大学智能科学与技术大三下的《智能系统课程设计》

课题的要求是用 Opencv 实现一个人脸识别系统，其中需要包含传统的 LBP 与 PCA 方法，同时还要使用 MineSpore 作为深度学习框架

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

本项目包含后端，后端仓库在另一边，之后再放出来

祝有需要的同学高分