# RL-GSBridge: 3D Gaussian Splatting Based Real2Sim2Real Method for Robotic Manipulation Learning


<font size=10> <p align="center"> **ICRA 2025** [[Paper](https://arxiv.org/abs/2409.20291)]</p></font>

Sim2real robot manipulation utilizing GS modeling

https://github.com/user-attachments/assets/4b1a21da-309e-41cf-9208-c315509055f8

Currently Released:
GS and mesh reconstruction
Code for policy training and GS rendering with Pybullet 

**TODO:** add running description; possible data; GS link?

## Environment Requirements
### For Real2Sim Reconstruction
Install SAM-Track as follows: ( ONLY for reconstruction of your own data.)
```bash
git clone https://github.com/z-x-yang/Segment-and-Track-Anything.git
cd Segment-and-Track-Anything && bash script/install.sh
bash script/download_ckpt.sh
```
Install the required dependencies of soft mesh binding GS as follows:
```bash
cd soft-gaussian-mesh-splatting
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
### For Sim2Real Policy Training
To run the sim2real policy, you just need to additionally install the pybullet package:
```bash
pip install pybullet
```
## Real2Sim Reconstruction
### An Example
We provide an example of GS training of 'Banana' in our paper. You could download data from [this link]( https://pan.baidu.com/s/1I1kX0oxD16T3Dfacwg6dkQ?pwd=at25), unpack the file and move it to folder **exp_obj_data**.

The mesh model and masks have been generated and preprocessed in this example. For full reconstruction from raw image data, please refer to the next section.
#### Soft Mesh Binding GS Reconstruction:
Create GS object folder and go to the folder **soft-gaussian-mesh-splatting**:
```bash
mkdir exp_obj_GS
cd soft-gaussian-mesh-splatting
```
For training, run:
```bash
python train.py -s ./exp_obj_data/banana/ -i mask_banana -m ./exp_obj_GS/banana_mesh_0 --gs_type gs_mesh_norm_aug --num_splats 2 --sh_degree 0
```
Here, 'gs_mesh_norm_aug' represents our soft mesh binding GS method. To run raw GaMeS method, change gs_type to 'gs_mesh'.

For evaluation of the training results, run:
```bash
# render the training set
python scripts/render.py --gs_type gs_mesh_norm_aug -m ./exp_obj_GS/banana_mesh_0
# calculate the metrics
python metrics.py --gs_type gs_mesh_norm_aug --model_paths ./exp_obj_GS/banana_mesh_0 
```
The evaluation results will be saved to file 'results_$GS_TYPE.json' under the model path.

Our training and evaluation results of soft mesh binding GS and the raw GaMeS are provided in [this link](https://pan.baidu.com/s/1EKa9_wKSu1NGgtkYVbhWUg?pwd=im84), where '_noaug' represents raw GaMeS training results.

#### Model Alignment:
**TODO**

### Reconstruct Your OWN DATA
To create the object mask and mesh models for your own data, SAM-Track and COLMAP are required. A recorded video of the object is needed.
#### Video Preprocessing:
**TODO**
#### Object Segmentation:
Follow the instruction in [SAM-Track](https://github.com/z-x-yang/Segment-and-Track-Anything) to segment each frame of your video. 
**TODO: FILE NAME DEFINITION**
#### Data Preprocessing:

## Sim2Real RL Policy Learning
### An Example
We provide some existing GS object and background models for 'Banana Grasping' training in [this link](https://pan.baidu.com/s/1T5BToLoAaTNn7F1BqWXjIw?pwd=qqu1). 

This file contains GS models of foam pad background and a cake, and their physical params are already recorded in **RLGS-bridge-pub/obj_trans.json**. Please unpack it and put the files into the **exp_obj_GS** folder.
#### Policy Training:
```bash
cd RLGS-bridge-pub
```
For policy training with GS rendering, run:
```bash
python learn_eih_SAC_meshGS.py -t 4 -q -b -c -i -l 'your training file path' -r -m mono --mesh --strain --color_refine --use_force
```
You could find your training logs under the folder **saves**. 

For training withour rendering, just run the code without  **-r**. For more params explaination, please refer to [this repo](https://github.com/IRMVLab/BCLearning).
#### Policy Test:
For policy test, run:
```bash
python test_eih_SAC_meshGS.py -t 4 -l 'your training file path' -b i -r --mesh
```
You could find the realistic rendering images under the folder **test_out**
# Acknowledgements
* Implementation of soft mesh binding GS inherited from [GaMeS](https://github.com/waczjoan/gaussian-mesh-splatting).
* Implementation of policy learning inherited from [DDPGwB](https://github.com/IRMVLab/BCLearning)

