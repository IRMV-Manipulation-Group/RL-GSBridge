import argparse
import os
import cv2
import skimage
import skimage
import numpy as np
import torch
import lpips
import colorama
from tqdm import tqdm

loss_fn = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

class MetricLogger:

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0
    def log(self, newvalue):
        self.sum += newvalue
        self.count += 1

    def avg(self):
        return self.sum/self.count

def parse_args():
    parser = argparse.ArgumentParser(description="计算指标")
    parser.add_argument("--gt", default="/root/autodl-tmp/train/ours_40000/re_textures", help="Ground Truth 所在的文件夹")
    parser.add_argument("--r", default="/root/autodl-tmp/train/ours_40000/renders", help="Rendered Views 所在的文件夹")
    return parser.parse_args()

def calc_psnr(img1, img2):
    if (img1 == img2).all():
        print(colorama.Fore.RED+ f"PSNR Warning: The input images are exactly the same. Returning 100." + colorama.Style.RESET_ALL)
        return 100.0
    else:
        return skimage.metrics.peak_signal_noise_ratio(img1, img2)

def calc_ssim(img1, img2):
    return skimage.metrics.structural_similarity(img1, img2, channel_axis=2,data_range=1)

def calc_lpips(img1, img2):
    def normalize_negative_one(img):
        normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        return 2*normalized_input - 1
    
    img1 = normalize_negative_one(img1)
    img2 = normalize_negative_one(img2)
    img1 = torch.tensor(img1).permute([2,0,1]).unsqueeze(0)
    img2 = torch.tensor(img2).permute([2,0,1]).unsqueeze(0)
    lpips_score = loss_fn(img1, img2)
    return lpips_score.item()

def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

args = parse_args()

gt_dir = [f for f in os.listdir(args.gt) if f.endswith('.png')]
gt_names = sorted(gt_dir)

r1 = [f for f in os.listdir(args.r) if f.endswith('.png')]
render_names = sorted(r1)

assert len(gt_names) == len(render_names)

image_count = len(gt_names)
gt_images = []
for gt_name in gt_names:
    img =read_image(os.path.join(args.gt, gt_name))
    gt_images.append(img)

render_images = []
for render_name in render_names:
    img = read_image(os.path.join(args.r, render_name))
    render_images.append(img)

psnr_logger = MetricLogger()
ssim_logger = MetricLogger()
lpips_logger = MetricLogger()

for idx, (img1, img2) in tqdm(enumerate(zip(gt_images, render_images))):
    # print(f"Processing {gt_names[idx]} and {render_names[idx]}")
    #print(img1.shape)
    if not img1.shape[0] == img2.shape[0]:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    psnr_logger.log(calc_psnr(img1, img2))
    ssim_logger.log(calc_ssim(img1, img2))
    lpips_logger.log(calc_lpips(img1, img2))

print(f"Avg PSNR: {psnr_logger.avg()}")
print(f"Avg SSIM: {ssim_logger.avg()}")
print(f"Avg LPIPS: {lpips_logger.avg()}")