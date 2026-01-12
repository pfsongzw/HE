import tenseal as ts
import numpy as np

print(">>> [模块] 高维数据：CKKS Tensor 操作演示 ")

# ==============================================================================
# 1. 环境准备 (Context Setup)
# ==============================================================================
# 定义 Degree
degree = 8192

ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=degree,
    coeff_mod_bit_sizes=[60, 40, 40, 60] # Depth = 2
)
ctx.global_scale = 2**40
ctx.auto_relin = True
ctx.auto_rescale = True
ctx.generate_galois_keys() # Tensor 运算必备

# 修复：手动计算 Slots
max_slots = degree // 2
print(f"✅ 环境配置完成: Max Slots={max_slots}")


# ==============================================================================
# 2. 创建加密张量 (Creating Tensors)
# ==============================================================================
print("\n--- A. 张量创建与基础属性 ---")

# 模拟一张 2x4 的微型图片 (像素值 0-255)
# Shape: (Height=2, Width=4)
plain_image = [
    [100, 150, 200, 250],
    [ 50,  60,  70,  80]
]

# 1. 从 List 创建
enc_tensor = ts.ckks_tensor(ctx, plain_image)

# 2. 查看形状 (属性)
print(f"原始形状 (Shape): {enc_tensor.shape}")

# 3. 基础运算：调整亮度 (所有像素 + 50)
# 自动广播 (Broadcasting)
enc_bright = enc_tensor + 50

print(f"原始数据:\n{np.array(plain_image)}")
# decrypt() 返回的是 Tensor 结构，可以转为 list 或 numpy
print(f"亮度调整后:\n{np.array(enc_bright.decrypt().tolist())}")


# ==============================================================================
# 3. 形状变换 (Reshape / Flatten)
# ==============================================================================
print("\n--- B. 形状变换 (Flatten) ---")
# 场景：CNN 的卷积层输出是 2D 图片，进入全连接层前需要展平 (Flatten)。

# 目标：将 (2, 4) 展平为 (1, 8)
# Reshape 极其快，因为它只是修改了元数据，没有执行加密计算
enc_flattened = enc_tensor.reshape([1, 8])

print(f"展平后形状: {enc_flattened.shape}")
# 注意：decrypt().tolist() 出来可能是多维列表，这里为了展示铺平效果
print(f"展平后数据: {enc_flattened.decrypt().tolist()}")


# ==============================================================================
# 4. 图像遮罩 (Image Masking / Filtering)
# ==============================================================================
print("\n--- C. 图像遮罩 (ROI 提取) ---")
# 场景：只保留图片左半部分，将右半部分置为 0 (加密状态下操作)

# 1. 准备遮罩 (Mask)
# 左边是 1 (保留)，右边是 0 (丢弃)
plain_mask = [
    [1, 1, 0, 0],
    [1, 1, 0, 0]
]

# 2. 应用遮罩 (Element-wise Multiplication)
# 密文图片 * 明文遮罩
# 消耗 1 层深度
enc_masked = enc_tensor * plain_mask

print(f"遮罩矩阵:\n{np.array(plain_mask)}")
print(f"遮罩后结果 (右侧应为0):\n{np.array(enc_masked.decrypt().tolist())}")


# ==============================================================================
# 5. 高级操作：方块求和 (Simple Pooling 模拟)
# ==============================================================================
print("\n--- D. 简易池化模拟 (Global Sum) ---")
# 计算整张图片的平均像素值

# sum() 会将 Tensor 中所有元素相加
# 结果仍然是一个加密对象
enc_total = enc_tensor.sum()

# 计算像素总数
num_pixels = 2 * 4

# 计算平均值
enc_avg = enc_total * (1 / num_pixels)

# decrypt() 出来可能是一个包含单个值的列表
print(f"像素总和: {enc_total.decrypt().tolist()}")
print(f"平均像素值: {enc_avg.decrypt().tolist()}")


# ==============================================================================
# 6. 多通道数据演示 (RGB)
# ==============================================================================
print("\n--- E. 多通道 RGB 数据 ---")
# Shape: (Channels=3, Height=2, Width=2)
rgb_data = [
    [[255, 0], [0, 255]], # R 通道
    [[0, 255], [255, 0]], # G 通道
    [[0, 0],   [100, 100]] # B 通道
]

enc_rgb = ts.ckks_tensor(ctx, rgb_data)
print(f"RGB 张量形状: {enc_rgb.shape}")

# 灰度化模拟: 统一对所有通道衰减 50%
enc_dimmed = enc_rgb * 0.5

print(f"R通道原始: {rgb_data[0]}")
# 解密后取第一个通道
decrypted_rgb = np.array(enc_dimmed.decrypt().tolist())
print(f"R通道变暗: \n{decrypted_rgb[0]}")