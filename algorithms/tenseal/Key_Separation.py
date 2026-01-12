import tenseal as ts
import os

# 定义模拟的文件存储路径 (在生产环境中，这对应网络发送)
KEY_DIR = "./tenseal_storage"
os.makedirs(KEY_DIR, exist_ok=True)

print(">>> [步骤 1] 客户端 (Alice): 初始化环境与密钥分离")

# 1. 创建完整的 Context (包含私钥 Secret Key)
# ---------------------------------------------------------
# Alice 在本地生成所有密钥。这是最高机密。
client_context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
client_context.global_scale = 2 ** 40
client_context.generate_galois_keys()
client_context.generate_relin_keys()

print(f"原始 Context 状态: 私钥={'✅' if client_context.has_secret_key() else '❌'}")

# 2. 序列化并保存“私钥上下文” (Client 本地备份)
# ---------------------------------------------------------
# save_secret_key=True 是关键。这串 bytes 必须锁在 Alice 的保险箱里。
secret_bytes = client_context.serialize(save_secret_key=True)

with open(f"{KEY_DIR}/alice_secret.ts", "wb") as f:
    f.write(secret_bytes)
print(f"🔒 [Client] 私钥上下文已保存 ({len(secret_bytes)} bytes)")

# 3. 生成并保存“公钥上下文” (发送给 Server)
# ---------------------------------------------------------
# 我们需要发给云端一个 Context，让他能做加法乘法，但不能解密。
# 方法 A: 仅序列化时排除私钥 (推荐)
public_bytes = client_context.serialize(save_secret_key=False)

# 方法 B: 在对象层面永久剥离私钥 (更彻底，用于防止内存泄漏)
# public_context_obj = client_context.copy()
# public_context_obj.make_context_public() # 这一步删除了内存中的私钥

with open(f"{KEY_DIR}/server_public.ts", "wb") as f:
    f.write(public_bytes)
print(f"🌍 [Client] 公钥上下文已发布 ({len(public_bytes)} bytes)")

# 4. 加密数据并序列化
# ---------------------------------------------------------
data = [10.0, 20.0, 30.0]
enc_vec = ts.ckks_vector(client_context, data)
enc_bytes = enc_vec.serialize()

with open(f"{KEY_DIR}/encrypted_data.ts", "wb") as f:
    f.write(enc_bytes)
print(f"📦 [Client] 数据已加密并打包 ({len(enc_bytes)} bytes)")

print("\n" + "=" * 50)
print("   🚧 网络传输边界 (Network Boundary) 🚧")
print("   假设 Alice 将 *.ts 文件发送给了云端 Bob")
print("=" * 50 + "\n")

print(">>> [步骤 2] 服务端 (Bob): 加载环境与盲算")

# 1. 加载公钥上下文
# ---------------------------------------------------------
# Bob 读取 server_public.ts。
with open(f"{KEY_DIR}/server_public.ts", "rb") as f:
    server_bytes = f.read()

# context_from: 从 bytes 恢复 Context 对象
server_context = ts.context_from(server_bytes)

# 关键安全检查：Bob 到底有没有私钥？
has_secret = server_context.has_secret_key()
print(f"🕵️ [Server] 检查权限: 是否拥有私钥? -> {'✅ 有 (危险!)' if has_secret else '❌ 无 (安全)'}")

# 2. 加载加密数据
# ---------------------------------------------------------
# 注意：恢复 ckks_vector 必须提供 context
with open(f"{KEY_DIR}/encrypted_data.ts", "rb") as f:
    data_bytes = f.read()

server_vec = ts.ckks_vector_from(server_context, data_bytes)

# 3. 尝试非法解密 (演示安全性)
# ---------------------------------------------------------
try:
    server_vec.decrypt()
except ValueError as e:
    print(f"🛡️ [Server] 尝试窃取数据失败: {e}")

# 4. 执行加密计算
# ---------------------------------------------------------
# Bob 虽然看不见数据，但可以对它进行计算
# 任务：计算 x^2 + 5
print("⚙️ [Server] 正在执行计算: x^2 + 5 ...")
result_vec = server_vec.square()
result_vec.add_(5)  # 原地加 5

# 5. 序列化结果并回传
# ---------------------------------------------------------
result_bytes = result_vec.serialize()
with open(f"{KEY_DIR}/result_data.ts", "wb") as f:
    f.write(result_bytes)
print(f"📤 [Server] 计算完成，结果已回传 ({len(result_bytes)} bytes)")

print("\n" + "=" * 50)
print("   🚧 网络传输边界 (Network Boundary) 🚧")
print("   Bob 将结果文件发回给 Alice")
print("=" * 50 + "\n")

print(">>> [步骤 3] 客户端 (Alice): 解密最终结果")

# 1. 恢复私钥上下文
# ---------------------------------------------------------
# Alice 从保险箱取出自己的 Context (带私钥)
with open(f"{KEY_DIR}/alice_secret.ts", "rb") as f:
    secret_bytes = f.read()

# 这里恢复出来的 Context 拥有解密能力
restore_client_context = ts.context_from(secret_bytes)

# 2. 加载结果数据
# ---------------------------------------------------------
with open(f"{KEY_DIR}/result_data.ts", "rb") as f:
    res_bytes = f.read()

# 使用带私钥的 Context 加载向量
final_vec = ts.ckks_vector_from(restore_client_context, res_bytes)

# 3. 解密与验证
# ---------------------------------------------------------
# 预期结果: 10^2+5=105, 20^2+5=405, 30^2+5=905
decrypted_vals = final_vec.decrypt()
print(f"🔓 [Client] 最终解密结果: {[round(v, 2) for v in decrypted_vals]}")

# 清理临时文件
import shutil

shutil.rmtree(KEY_DIR)
print("\n✅ 演示结束，临时文件已清理。")