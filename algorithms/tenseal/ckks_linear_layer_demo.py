import tenseal as ts
import numpy as np

print(">>> [模块] 神经网络核心：全连接层演示 ")

# ==============================================================================
# 1. 环境准备 (Context Setup) - 关键修复
# ==============================================================================
# 修复说明：
# 原代码只有 2 层深度，无法支撑 Linear -> Square -> Linear 的 3 次连续乘法。
# 这里我们配置了 4 层深度 (4个40)，并将 Degree 提升到 16384 以容纳这些参数。
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,  # ✅ 升级: 8192 -> 16384
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]  # ✅ 升级: Depth = 4
)
ctx.global_scale = 2 ** 40
ctx.auto_relin = True
ctx.auto_rescale = True

# ⚠️ [必须] 矩阵乘法涉及大量旋转，没有 Galois Keys 必报错
ctx.generate_galois_keys()

print(f"✅ 环境配置完成: Degree=16384, 支持深度=4 (完美支持多层网络)")

# ==============================================================================
# 2. 基础矩阵乘法演示 (Raw API)
# ==============================================================================
print("\n--- A. 基础 API 演示: x @ W + b ---")

# 场景：输入特征 3 个，输出特征 2 个
input_data = [0.5, -0.2, 0.8]

# 权重矩阵 (3x2)
plain_weight = [
    [1.0, 0.5],
    [-1.0, 0.5],
    [0.0, 1.0]
]

# 偏置向量 (2,)
plain_bias = [0.1, -0.1]

# 1. 加密输入
enc_x = ts.ckks_vector(ctx, input_data)

# 2. 执行矩阵乘法 (MatMul)
# 消耗 1 层深度
enc_matmul = enc_x.matmul(plain_weight)

# 3. 加偏置 (Bias Addition)
# 不消耗深度
enc_output = enc_matmul + plain_bias

# --- 验证 ---
np_x = np.array(input_data)
np_w = np.array(plain_weight)
np_b = np.array(plain_bias)
real_output = np_x.dot(np_w) + np_b

print(f"输入数据: {input_data}")
print(f"加密计算输出: {enc_output.decrypt()}")
print(f"Numpy 真实值: {real_output.tolist()}")

# ==============================================================================
# 3. 工程化封装：模拟 PyTorch 线性层 (Class Encapsulation)
# ==============================================================================
print("\n--- B. 进阶: 封装为 EncryptedLinear 类 ---")


class EncryptedLinear:
    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias

    def forward(self, enc_input):
        """
        前向传播: y = xW + b
        """
        # 1. 矩阵乘法 (消耗 1 Depth)
        out = enc_input.matmul(self.weight)

        # 2. 偏置加法 (不消耗 Depth)
        if self.bias is not None:
            out.add_(self.bias)  # 原地操作

        return out


# 模拟：5 输入 -> 3 输出
W_sim = np.random.uniform(-1, 1, size=(5, 3)).tolist()
b_sim = np.random.uniform(-0.5, 0.5, size=(3,)).tolist()

layer = EncryptedLinear(W_sim, b_sim)
in_data = [1.0, 2.0, 3.0, 4.0, 5.0]
enc_in = ts.ckks_vector(ctx, in_data)

enc_layer_out = layer.forward(enc_in)

print(f"层输入维度: 5")
print(f"层输出维度: 3")
print(f"前向传播结果: {enc_layer_out.decrypt()}")

# ==============================================================================
# 4. 完整网络推理链 (Chaining Layers) - 此前报错点
# ==============================================================================
print("\n--- C. 完整网络链演示: Linear -> Square -> Linear ---")

# Layer 1: 3 -> 3 (Identity)
W1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
b1 = [0, 0, 0]
layer1 = EncryptedLinear(W1, b1)

# Layer 2: 3 -> 1 (Sum features)
W2 = [[1], [1], [1]]
b2 = [0]
layer2 = EncryptedLinear(W2, b2)

# Input
x = ts.ckks_vector(ctx, [2.0, 3.0, 4.0])

print("1. 执行 Layer 1...")
h1 = layer1.forward(x)
# 深度消耗: 1 / 4

print("2. 执行 Activation (Square)...")
a1 = h1.square()
# 深度消耗: 2 / 4

print("3. 执行 Layer 2...")
# 原代码在此处报错，现在有足够的深度可以执行
out = layer2.forward(a1)
# 深度消耗: 3 / 4 (剩余 1 层缓冲)

# Logic check:
# x = [2, 3, 4] -> h1 = [2, 3, 4] -> a1 = [4, 9, 16] -> out = 4+9+16 = 29
result = out.decrypt()[0]

print("-" * 30)
print(f"多层网络预测值: {result:.4f}")
print(f"预期值: 29.0000")
print("-" * 30)