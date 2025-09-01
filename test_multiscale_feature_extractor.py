import torch
import pytest
from academic_implementation_toolkit import MultiScaleFeatureExtractor

def test_forward_shape():
    """测试正常输入下输出shape"""
    model = MultiScaleFeatureExtractor(input_dim=8, hidden_dim=16)
    x = torch.randn(4, 30, 8)  # (batch, seq_len, input_dim)
    out = model(x)
    assert out.shape == (4, 16)

def test_different_seq_len():
    """测试不同序列长度下的输出shape"""
    model = MultiScaleFeatureExtractor(input_dim=8, hidden_dim=16)
    for seq_len in [3, 10, 30, 50]:
        x = torch.randn(2, seq_len, 8)
        out = model(x)
        assert out.shape == (2, 16)

def test_window_larger_than_seq():
    """窗口大于序列长度时自动适配"""
    model = MultiScaleFeatureExtractor(input_dim=8, hidden_dim=16, short_window=10, medium_window=20, long_window=40)
    x = torch.randn(2, 5, 8)
    out = model(x)
    assert out.shape == (2, 16)

def test_gradient():
    """测试梯度反向传播"""
    model = MultiScaleFeatureExtractor(input_dim=8, hidden_dim=16)
    x = torch.randn(2, 30, 8, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_invalid_input_shape():
    """测试输入维度不符时抛出异常"""
    model = MultiScaleFeatureExtractor(input_dim=8, hidden_dim=16)
    x = torch.randn(2, 30, 7)  # input_dim应为8
    with pytest.raises(RuntimeError):
        model(x)

if __name__ == '__main__':
    pytest.main([__file__]) 