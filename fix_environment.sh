#!/bin/bash

echo "=== Fixing GPU Communication Environment ==="

# 设置正确的CUDA路径
export PATH="/apps/pkg/cuda/12.4/bin:$PATH"
export CUDA_HOME="/apps/pkg/cuda/12.4"
export LD_LIBRARY_PATH="/apps/pkg/cuda/12.4/lib64:$LD_LIBRARY_PATH"

# 禁用有问题的P2P通信
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# 增加超时时间
export NCCL_TIMEOUT=1800

# 调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

echo "Environment variables set:"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
echo "NCCL_TIMEOUT=$NCCL_TIMEOUT"

# 测试修复后的环境
python3 << TEST_EOF
import torch
import os

print("=== Testing Fixed Environment ===")
print(f"NCCL_P2P_DISABLE: {os.environ.get('NCCL_P2P_DISABLE', 'Not set')}")

# 测试数据传输
def test_transfer():
    # 使用CPU中转避免P2P问题
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # GPU 0
    with torch.cuda.device(0):
        gpu0 = data.cuda()
    
    # 通过CPU中转
    cpu_data = gpu0.cpu()
    with torch.cuda.device(1):
        gpu1 = cpu_data.cuda(1)
    
    # 返回验证
    final_data = gpu1.cpu().cuda(0)
    
    if torch.allclose(gpu0, final_data):
        print("✅ CPU-mediated transfer: SUCCESS")
        return True
    else:
        print("❌ CPU-mediated transfer: FAILED")
        return False

if test_transfer():
    print("Environment should be stable for training with P2P disabled")
else:
    print("Serious hardware issue detected - contact system administrator")
TEST_EOF
