import tenseal as ts
import numpy as np

print(">>> [模块] 线性代数与统计学基础演示 ")

# ==============================================================================
# 1. 环境准备 (Context Setup)
# ==============================================================================
# 修复核心：增加一层 40 bits 的冗余。
# 配置解析：[60, 40, 40, 40, 40, 60]
# - 顶层 60
# - 中间 4 个 40 (支持 4 次乘法，虽然方差只需要 3 次，但这多出的 1 次给了 Scale 缓冲空间)
# - 底层 60
# Degree 16384 支持约 438 bits，当前总和 280 bits，非常安全。

ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60] # ✅ 改动点：4个40
)
ctx.global_scale = 2**40
ctx.auto_relin = True
ctx.auto_rescale = True
ctx.generate_galois_keys()

print(f"✅ 环境配置完成: Degree=16384, 深度=4 (安全冗余模式)")


# ==============================================================================
# 2. 基础聚合：求和与平均
# ==============================================================================
print("\n--- A. 求和与平均值 ---")
data = [10.0, 20.0, 30.0, 40.0]
enc_vec = ts.ckks_vector(ctx, data)

enc_sum = enc_vec.sum()
enc_mean = enc_sum * (1 / len(data))

print(f"总和 (Sum): {enc_sum.decrypt()[0]:.2f}")
print(f"平均 (Mean): {enc_mean.decrypt()[0]:.2f}")


# ==============================================================================
# 3. 核心运算：点积
# ==============================================================================
print("\n--- B. 点积 (加权评分) ---")
features = [100.0, 500.0, 1000.0]
weights =  [0.5,   0.3,   0.2]
enc_features = ts.ckks_vector(ctx, features)

enc_score = enc_features.dot(weights)
print(f"点积评分: {enc_score.decrypt()[0]:.2f}")


# ==============================================================================
# 4. 高级统计：计算方差 (Variance)
# ==============================================================================
print("\n--- C. 计算方差 (终极测试) ---")
dataset = [1.0, 2.0, 3.0, 4.0, 5.0]
enc_data = ts.ckks_vector(ctx, dataset)
n_count = len(dataset)

print("1. 计算平均值...")
enc_mu = enc_data.sum() * (1 / n_count)

print("2. 计算差值...")
# 自动广播减法
enc_diff = enc_data - enc_mu

print("3. 计算平方...")
enc_sq_diff = enc_diff.square()

# 这里会消耗第 3 层深度。
# 由于我们现在有 4 层配置，这里有足够的模数空间容纳 Scale^2 的中间态
enc_variance = enc_sq_diff.sum() * (1 / n_count)

# 验证
real_var = np.var(dataset)
he_var = enc_variance.decrypt()[0]

print("-" * 30)
print(f"数据集: {dataset}")
print(f"加密计算方差: {he_var:.5f}")
print(f"Numpy 真实方差: {real_var:.5f}")
print(f"误差: {abs(he_var - real_var):.10f}")
print("-" * 30)