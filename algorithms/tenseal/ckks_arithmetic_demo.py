import tenseal as ts

print(">>> [模块] CKKS 基础与高阶算术演示 (v0.3.16 修正版)")

# ==============================================================================
# 1. 环境准备 (Context Setup)
# ==============================================================================
# Degree 16384 用于支持深度=3的计算 (如 x^3)
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
)
ctx.global_scale = 2**40
ctx.auto_relin = True
ctx.auto_rescale = True
ctx.generate_relin_keys()

print(f"✅ 环境配置完成: Degree=16384, 支持乘法深度=3")


# ==============================================================================
# 2. 基础四则运算
# ==============================================================================
vec_a = ts.ckks_vector(ctx, [10.0, 20.0, 30.0])
vec_b = ts.ckks_vector(ctx, [1.0,  2.0,  3.0])

print("\n--- A. 基础加减乘 ---")
print(f"加法 (Vec+Vec): {(vec_a + vec_b).decrypt()}")
print(f"广播 (Vec+100): {(vec_a + 100).decrypt()}")
print(f"减法 (Vec-Vec): {(vec_a - vec_b).decrypt()}")
print(f"乘法 (Vec*Vec): {(vec_a * vec_b).decrypt()}")


# ==============================================================================
# 3. 高阶数学运算 (Advanced Math) - 重点修复部分
# ==============================================================================
print("\n--- B. 高阶运算 (平方/幂/取反) ---")

vec_x = ts.ckks_vector(ctx, [2.0, 3.0])

# 1. 平方 (Square)
# .square() 依然存在且推荐使用
res_square = vec_x.square()
print(f"平方 (x^2): {res_square.decrypt()}")

# 2. 幂运算 (Power) - [已修正]
# 使用 Python 原生运算符 ** # 这会自动调用底层的 __pow__ 方法
try:
    # ❌ 旧语法: res_power = vec_x.power(3)
    # ✅ 新语法: 使用 ** 符号
    res_power = vec_x ** 3
    print(f"幂运算 (x^3): {res_power.decrypt()}")
except Exception as e:
    print(f"❌ 幂运算失败: {e}")

# 3. 取反 (Negation)
res_neg = -vec_x
print(f"取反 (-x): {res_neg.decrypt()}")


# ==============================================================================
# 4. 原地操作 (In-place Operations)
# ==============================================================================
print("\n--- C. 原地操作 (Memory Opt) ---")

large_vec = ts.ckks_vector(ctx, [100.0])

large_vec.add_(50)  # +50
large_vec.mul_(2)   # *2

# 如果想做原地幂运算，可以使用 pow_
large_vec.pow_(2)   # 原地平方 (300^2 = 90000)

print(f"原地计算结果: {large_vec.decrypt()}")


# ==============================================================================
# 5. 多项式求值 (Polyval)
# ==============================================================================
print("\n--- D. 多项式求值 ---")
# y = 1 + 2x + x^2
x_val = ts.ckks_vector(ctx, [5.0])
coeffs = [1, 2, 1]
poly_res = x_val.polyval(coeffs)
print(f"多项式结果 (x=5): {poly_res.decrypt()}")