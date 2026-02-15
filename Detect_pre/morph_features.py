import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

def extract_morph_features(heatmap, threshold=0.7, multi_scale=False):
    """
    从热力图中提取形态学特征
    
    Args:
        heatmap: 2D numpy数组 (0-1范围)
        threshold: 二值化阈值，默认0.7
        multi_scale: 是否进行多尺度提取，默认False
    
    Returns:
        max_features: 最大连通域的特征字典
                     包含: area, hull_area, convexity, eccentricity, 
                           centroid_x, centroid_y, entropy
        all_features: 所有连通域的特征列表
    """
    
    def _calculate_entropy(img):
        """计算图像信息熵"""
        img_normalized = (img * 255).astype(np.uint8)
        hist, _ = np.histogram(img_normalized, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        return entropy
    
    def _calculate_eccentricity(coords):
        """计算离心率（基于主轴）"""
        if len(coords) < 2:
            return 0.0
        
        # 计算协方差矩阵
        covariance = np.cov(coords.T)
        
        # 处理一维情况
        if covariance.ndim == 0:
            return 0.0
        
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # 离心率 = sqrt(1 - (小轴²/大轴²))
        if eigenvalues[0] > 1e-10:
            eccentricity = np.sqrt(max(0, 1 - (eigenvalues[1] / eigenvalues[0])))
        else:
            eccentricity = 0.0
        
        return float(eccentricity)
    
    def _extract_features_single(img):
        """单尺度特征提取"""
        # 二值化
        binary = img > threshold
        
        # 标记连通域
        labeled, num_features = ndimage.label(binary)
        
        # 计算热力图熵
        entropy = _calculate_entropy(img)
        
        all_features = []
        max_area = 0
        max_feature_dict = None
        
        # 遍历每个连通域
        for label_idx in range(1, num_features + 1):
            region = (labeled == label_idx)
            
            # 面积
            area = np.sum(region)
            
            # 获取区域坐标
            coords = np.argwhere(region)
            
            if len(coords) < 3:  # 凸包需要至少3个点
                continue
            
            # 凸包面积和凸包比
            try:
                hull = ConvexHull(coords)
                hull_area = hull.volume  # 2D中为面积
                convexity = area / hull_area if hull_area > 0 else 0
            except ValueError:
                hull_area = float(area)
                convexity = 1.0
            
            # 离心率
            eccentricity = _calculate_eccentricity(coords)
            
            # 质心 (行列坐标转换为x,y)
            center_row, center_col = ndimage.center_of_mass(region)
            centroid_x = float(center_col)
            centroid_y = float(center_row)
            
            # 构建特征字典
            feature_dict = {
                'area': float(area),
                'hull_area': float(hull_area),
                'convexity': float(convexity),
                'eccentricity': float(eccentricity),
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'entropy': float(entropy)
            }
            
            all_features.append(feature_dict)
            
            # 记录最大面积的区域
            if area > max_area:
                max_area = area
                max_feature_dict = feature_dict.copy()
        
        # 若无有效连通域，返回空特征
        if max_feature_dict is None:
            max_feature_dict = {
                'area': 0.0,
                'hull_area': 0.0,
                'convexity': 0.0,
                'eccentricity': 0.0,
                'centroid_x': 0.0,
                'centroid_y': 0.0,
                'entropy': float(entropy)
            }
        
        return max_feature_dict, all_features
    
    # 单尺度提取
    if not multi_scale:
        return _extract_features_single(heatmap)
    
    # 多尺度提取 (0.8, 1.0, 1.2)
    scales = [0.8, 1.0, 1.2]
    max_features_list = []
    all_features_combined = []
    
    h, w = heatmap.shape
    
    for scale in scales:
        # 缩放图像
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_heatmap = ndimage.zoom(heatmap, scale, order=1)
        
        max_feat, all_feat = _extract_features_single(scaled_heatmap)
        
        # 质心坐标还原到原始尺度
        max_feat_adjusted = max_feat.copy()
        max_feat_adjusted['centroid_x'] *= (1.0 / scale)
        max_feat_adjusted['centroid_y'] *= (1.0 / scale)
        max_feat_adjusted['scale'] = scale
        
        max_features_list.append(max_feat_adjusted)
        all_features_combined.extend(all_feat)
    
    # 合并最大特征（取均值）
    merged_max = {}
    feature_keys = ['area', 'hull_area', 'convexity', 'eccentricity', 
                     'centroid_x', 'centroid_y', 'entropy']
    
    for key in feature_keys:
        values = [feat[key] for feat in max_features_list]
        merged_max[key] = float(np.mean(values))
    
    merged_max['scales'] = scales
    
    return merged_max, all_features_combined