#! /bin/bash  
# employ bash shell

# /data/exp_obj_data/banana_sharp.rar
# /data/exp_obj_data/banana/

OBJECT="bear" # "bg_plate_new" # "bg_tablecloth" "bg_colorful_mat" "bg_large_plate" "bg_mid_plate" "bg_small_plate"

RAR_PATH="/data/exp_obj_data/${OBJECT}_sharp.rar"
OUT_SHARP_PATH="/data/exp_obj_data/${OBJECT}/${OBJECT}_sharp"
OUT_PATH="/data/exp_obj_data/${OBJECT}"

if [[ ! -d "$OUT_PATH" ]]; then
    mkdir -p "$OUT_SHARP_PATH"
    echo "创建子目录：$OUT_SHARP_PATH"
fi

unrar x -o- -y "$RAR_PATH"  "$OUT_PATH"
