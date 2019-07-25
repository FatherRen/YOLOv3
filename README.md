# YOLOv3
## 之前做过的道路车辆检测项目的简化版代码

## 1、介绍：
### 主目录下包含6个文件：
      model：保存训练的模型
      summary：保存日志文件
      test_data：保存测试数据
      train_data：保存训练数据
                  训练数据不能提供，可以自己制作数据集
                  数据集的格式为：
                      图像名称.jpg,左上x_左上y_右下x_右下y_class；...;左上x_左上y_右下x_右下y_class
      config：模型配置
      main：主函数
      
## 2、使用：
      （1）训练：python main.py --is_training True
      （2）测试：python main.py --is_training False --data_path "测试数据路径.jpg"
      
## 3、注意：
      本代码在训练的时候只能在Ubuntu系统下运行，在Windows下会报显存不够用的错误，2070显卡不知所措。。。
