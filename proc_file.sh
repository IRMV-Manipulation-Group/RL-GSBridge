#! /bin/bash  # employ bash shell

# /data/exp_obj_data/banana_sharp.rar
# /data/exp_obj_data/banana/

OBJECT="large_cube_purple" # "banana" "toycat" "gift" 

RAR_PATH="/data/exp_obj_data/${OBJECT}_sharp.rar"
OUT_SHARP_PATH="/data/exp_obj_data/${OBJECT}/${OBJECT}_sharp"
OUT_PATH="/data/exp_obj_data/${OBJECT}"

if [[ ! -d "$OUT_PATH" ]]; then
    mkdir -p "$OUT_SHARP_PATH"
    echo "创建子目录：$OUT_SHARP_PATH"
fi

unrar x -o- -y "$RAR_PATH"  "$OUT_PATH"

MASK_PATH="/data/Segment-and-Track-Anything/tracking_results/${OBJECT}_sharp/${OBJECT}_sharp_masks"

cp -r "$MASK_PATH" "$OUT_PATH"

python /data/exp_obj_data/mask_proc.py -n "$OBJECT"
