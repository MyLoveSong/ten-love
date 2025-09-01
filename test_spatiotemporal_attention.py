import torch
import pytest
from academic_implementation_toolkit import SpatioTemporalAttention

def test_forward_shape():
    """测试正常输入下输出shape"""
    model = SpatioTemporalAttention(input_dim=8, num_heads=2)
    x = torch.randn(8, 12, 8)  # (batch, seq_len, input_dim)
    out = model(x)
    assert out.shape == (8, 12, 8)

def test_different_batch_seq():
    """测试不同batch/seq_len下的输出shape"""
    model = SpatioTemporalAttention(input_dim=8, num_heads=2)
    for batch in [1, 4]:
        for seq_len in [5, 20]:
            x = torch.randn(batch, seq_len, 8)
            out = model(x)
            assert out.shape == (batch, seq_len, 8)

def test_gradient():
    """测试梯度反向传播"""
    model = SpatioTemporalAttention(input_dim=8, num_heads=2)
    x = torch.randn(2, 6, 8, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_fusion_type_add():
    """测试融合方式为add时的行为"""
    model = SpatioTemporalAttention(input_dim=8, num_heads=2, fusion_type='add')
    x = torch.randn(3, 7, 8)
    out = model(x)
    assert out.shape == (3, 7, 8)  # add为Identity，输出shape应与输入一致

def test_invalid_fusion_type():
    """测试不支持的融合方式抛出异常"""
    with pytest.raises(ValueError):
        SpatioTemporalAttention(input_dim=8, num_heads=2, fusion_type='unknown')

def test_invalid_input_shape():
    """测试输入维度不符时抛出异常"""
    model = SpatioTemporalAttention(input_dim=8, num_heads=2)
    x = torch.randn(2, 5, 7)  # input_dim应为8
    with pytest.raises(AssertionError):
        model(x)

if __name__ == '__main__':
    pytest.main([__file__]) 