import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib

feat_dim = 16#16 # 384
def vis_feat(feat, path, patch_w):
    features = torch.zeros(4, patch_w * patch_w, feat_dim)
    features[0] = feat
    # 重塑特征形状为(4 * patch_h * patch_w, feat_dim)
    features = features.reshape(4 * patch_w * patch_w, feat_dim).cpu()

    # 创建PCA对象并拟合特征
    pca = PCA(n_components=3)
    pca.fit(features)

    # 对PCA转换后的特征进行归一化处理
    pca_features = pca.transform(features)
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())

    # 根据阈值进行前景和背景的区分
    pca_features_fg = pca_features[:, 0] > 0.3
    pca_features_bg = ~pca_features_fg

    # 查找背景特征的索引
    b = np.where(pca_features_bg)

    # 对前景特征再次进行PCA转换
    pca.fit(features[pca_features_fg])
    pca_features_rem = pca.transform(features[pca_features_fg])

    # 对前景特征进行归一化处理
    for i in range(3):
        pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
        # 使用均值和标准差进行转换，个人发现这种转换方式可以得到更好的可视化效果
        # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

    # 创建RGB特征数组
    pca_features_rgb = pca_features.copy()

    # 替换前景特征为转换后的特征
    pca_features_rgb[pca_features_fg] = pca_features_rem

    # 将背景特征设置为0
    pca_features_rgb[b] = 0

    # 重塑特征形状为(4, patch_h, patch_w, 3)
    pca_features_rgb = pca_features_rgb.reshape(4, patch_w, patch_w, 3)

    # 显示第一个图像的RGB特征
    plt.imshow(pca_features_rgb[0][...,::-1])
    plt.savefig(path)

