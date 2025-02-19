#! /bin/bash  # employ bash shell


OBJECT="bg_plate_new" #"bg_raw" "bg_tablecloth" "bg_colorful_mat" "bg_large_plate" "bg_mid_plate" "bg_small_plate"

DENSE_PATH="/data/exp_obj_data/${OBJECT}/dense"
OUTPUT_PATH="/data/exp_obj_data/${OBJECT}/scene.mvs"
IMAGE_FOLDER="/data/exp_obj_data/${OBJECT}/dense/images"

/data/openMVS_build/bin/InterfaceCOLMAP -i "$DENSE_PATH" -o "$OUTPUT_PATH" --image-folder "$IMAGE_FOLDER"

# /data/openMVS_build/bin/InterfaceCOLMAP -i /data/exp_obj_data/banana/dense -o /data/exp_obj_data/banana/scene.mvs --image-folder /data/exp_obj_data/banana/dense/images

OBJECT_PATH="/data/exp_obj_data/${OBJECT}"

CUDA_VISIBLE_DEVICES=2 /data/openMVS_build/bin/DensifyPointCloud "$OUTPUT_PATH" -w "$OBJECT_PATH"

# /data/openMVS_build/bin/DensifyPointCloud /data/exp_obj_data/banana/scene.mvs -w /data/exp_obj_data/banana

SCENE_DENSE_PATH="/data/exp_obj_data/${OBJECT}/scene_dense.mvs"
PLY_PATH="/data/exp_obj_data/${OBJECT}/scene_dense.ply"

/data/openMVS_build/bin/ReconstructMesh "$SCENE_DENSE_PATH" -p "$PLY_PATH" -w "$OBJECT_PATH"

# /data/openMVS_build/bin/ReconstructMesh /data/exp_obj_data/banana/scene_dense.mvs -p /data/exp_obj_data/banana/scene_dense.ply -w /data/exp_obj_data/banana

DENSE_MESH_PLY_PATH="/data/exp_obj_data/${OBJECT}/scene_dense_mesh.ply"
MESH_REFINE_PATH="/data/exp_obj_data/${OBJECT}/scene_dense_mesh_refine.mvs"

/data/openMVS_build/bin/RefineMesh "$SCENE_DENSE_PATH" -m "$DENSE_MESH_PLY_PATH" -w "$OBJECT_PATH" -o "$MESH_REFINE_PATH" --scales 1 --max-face-area 16

# /data/openMVS_build/bin/RefineMesh /data/exp_obj_data/banana/scene_dense.mvs -m /data/exp_obj_data/banana/scene_dense_mesh.ply -w /data/exp_obj_data/banana -o /data/exp_obj_data/banana/scene_dense_mesh_refine.mvs --scales 1 --max-face-area 16

REFINE_PLY_PATH="/data/exp_obj_data/${OBJECT}/scene_dense_mesh_refine.ply"
REFINE_TEXTURE_PATH="/data/exp_obj_data/${OBJECT}/scene_dense_mesh_refine_texture.mvs"

/data/openMVS_build/bin/TextureMesh "$SCENE_DENSE_PATH" -m "$REFINE_PLY_PATH" -w "$OBJECT_PATH" -o "$REFINE_TEXTURE_PATH"

# /data/openMVS_build/bin/TextureMesh  /data/exp_obj_data/banana/scene_dense.mvs -m /data/exp_obj_data/banana/scene_dense_mesh_refine.ply -w /data/exp_obj_data/banana -o /data/exp_obj_data/banana/scene_dense_mesh_refine_texture.mvs

DMAP_PATH="/data/exp_obj_data/${OBJECT}/*.dmap"
rm $DMAP_PATH

# python /data/exp_obj_data/fix_mesh.py -n "$OBJECT"

# mesh模型不存在空洞，如果在填补空洞处报错，直接运行mesh生成
python /data/exp_obj_data/gen_mesh.py -n "$OBJECT"
