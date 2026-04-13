import logging
import numpy as np
import cv2

def RGB2YUV(x_rgb):
    # 使用 float 替代 np.float 适配 NumPy 2.0
    x_yuv = np.zeros(x_rgb.shape, dtype=float)
    for i in range(x_rgb.shape[0]):
        # 确保输入是 uint8 且范围在 0-255
        img_uint8 = np.clip(x_rgb[i], 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
        x_yuv[i] = img
    return x_yuv

def YUV2RGB(x_yuv):
    x_rgb = np.zeros(x_yuv.shape, dtype=float)
    for i in range(x_yuv.shape[0]):
        img_uint8 = np.clip(x_yuv[i], 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img_uint8, cv2.COLOR_YCrCb2RGB)
        x_rgb[i] = img
    return x_rgb

def DCT(x_train, window_size):
    # x_train: (idx, w, h, ch) -> 转置为 (idx, ch, w, h) 方便处理
    x_dct = np.zeros((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]), dtype=float)
    x_train_trans = np.transpose(x_train, (0, 3, 1, 2))

    for i in range(x_train_trans.shape[0]):
        for ch in range(x_train_trans.shape[1]):
            for w in range(0, x_train_trans.shape[2], window_size):
                for h in range(0, x_train_trans.shape[3], window_size):
                    # DCT 变换
                    sub_block = x_train_trans[i][ch][w:w+window_size, h:h+window_size].astype(float)
                    sub_dct = cv2.dct(sub_block)
                    x_dct[i][ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct

def IDCT(x_train, window_size):
    # x_train: (idx, ch, w, h)
    x_idct = np.zeros(x_train.shape, dtype=float)
    for i in range(x_train.shape[0]):
        for ch in range(0, x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    # 逆 DCT 变换
                    sub_block = x_train[i][ch][w:w+window_size, h:h+window_size].astype(float)
                    sub_idct = cv2.idct(sub_block)
                    x_idct[i][ch][w:w+window_size, h:h+window_size] = sub_idct
    # 转回 (idx, w, h, ch)
    x_idct = np.transpose(x_idct, (0, 2, 3, 1))
    return x_idct

class ftrojann_version(object):
    def __init__(self, YUV, channel_list, window_size, magnitude, pos_list) -> None:
        self.YUV = YUV
        self.channel_list = channel_list
        self.window_size = window_size
        self.magnitude = magnitude
        self.pos_list = pos_list

    def __call__(self, img):
        # 1. 预处理：将单张图 (W, H, C) 扩展为 (1, W, H, C)
        img = np.expand_dims(img, axis=0)
        
        if self.YUV:
            img = RGB2YUV(img)

        # 2. 变换到频域
        img_freq = DCT(img, self.window_size)  # 得到 (idx, ch, w, h)

        # 3. 植入触发器（坐标解析鲁棒性修复）
        for i in range(img_freq.shape[0]):
            for ch in self.channel_list:
                for w in range(0, img_freq.shape[2], self.window_size):
                    for h in range(0, img_freq.shape[3], self.window_size):
                        # --- 核心修复逻辑：防范 pos_list 格式混乱 ---
                        for pos in self.pos_list:
                            try:
                                # 尝试处理 [[15, 15]] 或 [(15, 15)] 格式
                                if isinstance(pos, (list, tuple, np.ndarray)):
                                    p_w, p_h = int(pos[0]), int(pos[1])
                                else:
                                    # 如果 pos 只是一个数字，说明 pos_list 被解析成了 [15, 15]
                                    # 我们直接取 pos_list 的前两个值作为坐标并跳出循环
                                    p_w, p_h = int(self.pos_list[0]), int(self.pos_list[1])
                                    img_freq[i][ch][w + p_w][h + p_h] += self.magnitude
                                    break 
                                
                                img_freq[i][ch][w + p_w][h + p_h] += self.magnitude
                            except Exception:
                                # 最后的兜底方案：万一解析全错，强制使用固定坐标 (15, 15)
                                img_freq[i][ch][w + 15][h + 15] += self.magnitude
                                break

        # 4. 逆变换回空间域
        img_idct = IDCT(img_freq, self.window_size)

        if self.YUV:
            img_idct = YUV2RGB(img_idct)
            
        # 5. 后处理：去除 batch 维度，返回 (W, H, C)
        img_final = np.squeeze(img_idct, axis=0)
        # 确保像素值在合理范围内
        img_final = np.clip(img_final, 0, 255).astype(np.uint8)
        return img_final