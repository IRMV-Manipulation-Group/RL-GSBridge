import cv2
import numpy as np
import os
import argparse
def extract_number(filename):
    number = ''
    for char in filename:
        if char.isdigit():
            number += char
    return int(number)

def sort_filenames(folder_path):
    filenames = os.listdir(folder_path)
    filenames = sorted(filenames, key=lambda x: extract_number(x))
    return filenames

def max_mask(mask):
    maskcop = mask[:, :, 0]
    # 单通道轮廓
    contours, _ = cv2.findContours(maskcop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 排序面积
    #contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    contours=sorted(contours,key=lambda c:cv2.contourArea(c),reverse=True)
    #---------------------------
    background = np.zeros_like(mask)
    #第三个参数是contour的索引，第四个参数是RGB颜色设置，最后的参数-1表示填充，其他表示用对应粗细的线条绘制轮廓
    biggest_contour_mask = cv2.drawContours(background, contours, 0, (255, 255, 255), -1)
    ##这里再还原到单通道，方便我们后续处理
    #biggest_contour_mask = cv2.cvtColor(biggest_contour_mask, cv2.COLOR_RGB2GRAY)

    return biggest_contour_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='')
    args = parser.parse_args()
    # data_path = "./cake/cake1_sparse/"
    # mask_path = "./cake/output_cake_masks/"
    # out_path = './cake/mask_cake/'
    # masko_path = './cake/mask/'

    data_path = "/data/exp_obj_data/"+ args.name +"/"+ args.name +"_sharp/"
    mask_path = "/data/exp_obj_data/"+ args.name +"/"+ args.name +"_sharp_masks/"
    out_path = "/data/exp_obj_data/"+ args.name +"/mask_"+ args.name +"/"
    masko_path = "/data/exp_obj_data/"+ args.name +"/mask/"
    masko_path_w = "/data/exp_obj_data/"+ args.name +"/white_mask_"+ args.name +"/"

    if not os.path.exists(masko_path):
        os.makedirs(os.path.dirname(masko_path), exist_ok=True)
    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(masko_path_w):
        os.makedirs(os.path.dirname(masko_path_w), exist_ok=True)

    print(data_path)
    
    count=0
    values = None
    dir_list = sort_filenames(data_path)
    #print(dir_list)
    num = len(dir_list)
    for i in range(num):
        image_name = dir_list[i]
        image_path = data_path + dir_list[i]
        #print(image_path)
        #image_path = data_path + f"img_{i}.png"
        maskim_path = mask_path + '%05d.png'%count
        img = cv2.imread(image_path)
        mask = cv2.imread(maskim_path)

        if values is None:
            mask_n = np.array(mask)
            #print(mask_n.shape)
            unique_values1 = list(set(mask_n[:, :, 0].reshape(-1).tolist()))
            unique_values2 = list(set(mask_n[:, :, 1].reshape(-1).tolist()))
            unique_values3 = list(set(mask_n[:, :, 2].reshape(-1).tolist()))
            seg_num = len(unique_values1) - 1
            values = []
            
            for i in range(len(unique_values1)-1):
                values.append(mask_n[mask_n[:, :, 0]==unique_values1[i+1]][0])
            #non_zero_values = np.nonzero(mask_n)
            # 打印非零值
            #value_1 = mask_n[non_zero_values[0][0], non_zero_values[1][0]]
            print("mask value:", values)

        mask_all = np.zeros_like(mask)[:, :, 0]
        for j, value in enumerate(values):
            #print("mask value:", value)
            img_new = img.copy()
            img_new_w = img.copy()
            mask_new = np.ones_like(mask)*255
            #print(np.where(mask[:, :, 0]==value[0]))
            mask_new[mask[:, :, 0]!=value[0]] = 0
            mask_new[mask[:, :, 1]!=value[1]] = 0
            mask_new[mask[:, :, 2]!=value[2]] = 0
            mask_proc = max_mask(mask_new)
            #mask_proc = mask_new

            img_new[mask_proc!=255] = 0
            img_new_w[mask_proc!=255] = 255
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(out_path + f"img_obj{j}_{i}.png", img_new)
            cv2.imwrite(out_path + image_name, img_new)
            cv2.imwrite(masko_path_w + image_name, img_new_w)
            #non_zero_values = np.nonzero(img)
            #img[non_zero_values[0], non_zero_values[1]]=255 ### binary mask

            #cv2.imwrite(masko_path + image_name, img)
            mask_all += (mask_proc[:,:,0]/255).astype(np.uint8)*(j+1)
        #print(set(list(mask_all.reshape(-1))))
        np.save(masko_path + image_name.replace('.png', ''), mask_all)
        count+=1


    #cv2.destroyAllWindows()
    print(f"The masks of <{args.name}> have been done!")


if __name__ == "__main__":
    main()