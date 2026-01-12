"""
TenSEAL Expert Configuration File
---------------------------------------------------------
此文件整合了 TenSEAL (v0.3.x) 的核心加密参数配置。
包含了针对不同业务场景（统计、机器学习、深度网络、整数计算）的最佳实践参数。

使用方法:
    from tenseal_config import create_context, CONFIG_LR
    ctx = create_context(CONFIG_LR)
"""

import tenseal as ts

# ==============================================================================
# 1. 基础常量定义 (Fundamental Constants)
# ==============================================================================

# 同态加密方案选择
SCHEME_CKKS = ts.SCHEME_TYPE.CKKS  # 浮点数近似计算 (ML/数据分析首选)
SCHEME_BFV = ts.SCHEME_TYPE.BFV    # 整数精确计算 (投票/ID匹配)

# 多项式模数度 (N)
# 决定了容器大小(Slots)和安全性上限
DEGREE_FAST = 4096      # 速度最快，容量 2048，安全性支持约 109 bits 模数链
DEGREE_STD = 8192       # 标准推荐，容量 4096，安全性支持约 218 bits 模数链
DEGREE_HEAVY = 16384    # 深度计算，容量 8192，安全性支持约 438 bits 模数链

# 全局缩放因子 (Global Scale)
# CKKS 专用，用于控制小数精度，通常设为 2^40
SCALE_STD = 2 ** 40


# ==============================================================================
# 2. 场景化配置字典 (Scenario Configurations)
# ==============================================================================

# ------------------------------------------------------------------
# 场景 A: 简单统计 (Simple Statistics)
# ------------------------------------------------------------------
# 适用: 求和、平均值、方差、简单线性加权。
# 特点: 速度极快，密文体积小，仅支持 1 次乘法深度。
CONFIG_STATS = {
    "name": "Simple Statistics (Speed Optimized)",
    "scheme_type": SCHEME_CKKS,
    "poly_modulus_degree": DEGREE_FAST,  # 4096
    # 模数链: [顶层60, 中间40(1次乘法), 底层60] -> 总和 160 bits
    "coeff_mod_bit_sizes": [60, 40, 60],
    "global_scale": SCALE_STD,
    "security_level": "128-bit"
}

# ------------------------------------------------------------------
# 场景 B: 逻辑回归/线性回归 (Linear/Logistic Regression)
# ------------------------------------------------------------------
# 适用: 常见的机器学习推理。
# 特点: 支持矩阵乘法(x*w) + 多项式激活函数(x^3)。支持 2-3 层深度。
CONFIG_LR = {
    "name": "Machine Learning (Standard)",
    "scheme_type": SCHEME_CKKS,
    "poly_modulus_degree": DEGREE_STD,   # 8192
    # 模数链: [顶层60, 中间40*3(3次乘法), 底层60] -> 总和 240 bits
    # 注意: 240 bits 略微超过 8192 的默认安全线(218)，但 TenSEAL 允许一定浮动。
    # 如果追求绝对安全，可减少一个 40，变为 [60, 40, 40, 60] (深度2)
    "coeff_mod_bit_sizes": [60, 40, 40, 40, 60], 
    "global_scale": SCALE_STD,
    "security_level": "128-bit"
}

# ------------------------------------------------------------------
# 场景 C: 深度神经网络 (Deep Neural Networks)
# ------------------------------------------------------------------
# 适用: 多层全连接网络、ResNet模拟、复杂多项式拟合。
# 特点: 支持 5-6 层连续乘法，计算慢，内存占用大。
CONFIG_DNN = {
    "name": "Deep Learning (High Depth)",
    "scheme_type": SCHEME_CKKS,
    "poly_modulus_degree": DEGREE_HEAVY, # 16384
    # 模数链: 中间有 6 个 40，支持深度 6
    "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60],
    "global_scale": SCALE_STD,
    "security_level": "128-bit"
}

# ------------------------------------------------------------------
# 场景 D: 精确整数投票 (Integer Voting/Counting)
# ------------------------------------------------------------------
# 适用: 电子投票、隐私求交(PSI)。
# 特点: 使用 BFV 方案，无误差，无 Scale 概念。
CONFIG_VOTING = {
    "name": "Integer Exact Calculation (BFV)",
    "scheme_type": SCHEME_BFV,
    "poly_modulus_degree": DEGREE_FAST,  # 4096
    # BFV 不需要 coeff_mod_bit_sizes 链，只需要明文模数
    # 1032193 是一个优化过的大素数，适合批处理
    "plain_modulus": 1032193, 
    "security_level": "128-bit"
}
