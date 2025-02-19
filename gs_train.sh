#! /bin/bash

OBJECT="gift" # "banana" "toycat" "gift" "large_cube_purple" "bg_tablecloth"

DATA="/data/exp_obj_data/${OBJECT}/"
#IMAGES="mask_${OBJECT}"
IMAGES="mask_${OBJECT}"
MESH="/data/exp_obj_GS/${OBJECT}_mesh_0"

# CUDA_VISIBLE_DEVICES=3 python /data/gaussian-mesh-splatting/train.py -s $DATA -i $IMAGES -m $MESH --gs_type gs_mesh_norm_aug --num_splats 2
CUDA_VISIBLE_DEVICES=2 python /data/gaussian-mesh-splatting/train.py -s $DATA -i $IMAGES -m $MESH --gs_type gs_mesh_norm_aug --num_splats 2 --sh_degree 0 -w