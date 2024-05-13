2023/12/16 
运行python main.py --mode train --config ./configs/config_JointSSG_full_l160.yaml时候报错，分析发现是rio image生成的h5文件（hdf5格式）中存在部分错误，会报错读取问题
例如这个路径的h5文件，/home/ycb/3DSSG/./data/dataset//data/roi_images.h5/roi_images/a0905ffc-66f7-2272-9f05-966bab3ce8ad.h5


2024/1/12

目前遗留的坑：

**gt_edge完全由固定边生成的问题**

**Eval测试部分代码阅读，融合进获得box信息，和将信息记录进block**

**eval的方法，工具，结果**

2023/1/16

可以train了

目前遗留的坑：

**eval的方法，工具，结果**

踩过的坑：

在设计记忆模块的时候，保存tensor的时候需要.detach()确保将tensor其从计算图中分离，否则tensor依然会具有梯度信息，并且会在第二次backward过程中报错，
报错为重复的Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)

如果在交叉熵loss处，输入的gt类别超出了给出的类别，就会报错RuntimeError: CUDA error: device-side assert triggered


2023/1/19

顿悟

不能在confirm_rel函数内，计算max值，直接从这里返回box1和box2之间的9种关系的置信度得分tensor[9]，和box1与box2之间关系的gt tensor[1]即可，计算max不由这里决定
！！！！！！！