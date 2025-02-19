# KUKA机械臂仿真
## 运行环境需求
pybullet, numpy

## kuka.py
包含三个操作任务环境
1. KukaCamEnv1，物块放入杯子
2. KukaCamEnv2，物块堆叠
3. KukaCamEnv3，杯子插入杯子

## main.py
1. actor()
接收感知信息输出控制信号
2. test_actor()
测试actor性能


## 第一视角强化学习--Usage

新增了_eih文件，可以实现第一视角RL
### Train
Train the complete algorithm with state-input. `-t` is the training task. `-l` is the saving log.
```
python learn_eih.py -q -b -c -t 1 -l 1
```
Train the complete algorithm with eye-in-hand image-input. (modified by wyx)
#### TODO: 把两个分支的图片变成一个分支的。
```
OMP_NUM_THREADS=3 python learn_eih.py -q -b -c -i -l 3
```

### test
Test the model in log `-l` of task `-t`.
```
python test_eih.py -t 1 -l 2 -b i
```

### results
```
test success rate(%):
state input	    88.4
dual img input	49.0
eih img input	71.1
```
----------------------

# 3D Gaussian Editing
readme to be modified......
### 环境
文件在neural-scene-graphs-3dgs中， 有3DGS的环境即可运行（应该）
## 手动编辑
已实现：输入transform matrix可以编辑物体位置，现在用的是仿真背景与挖掘机进行的测试
```
python gs_rendering.py --white_background
```
### results
![image-editing](../LOG/editing.jpg)
### TODO：
1.载入仿真中单独建模的物体进行编辑测试
2.接口到pybullet测试
3.mobileSAM 分割

----------------------