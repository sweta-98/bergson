import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from bergson.utils.utils import convert_dtype_to_np, tensor_to_numpy


class TestConvertDtypeToNp:
    def test_float16(self):
        assert convert_dtype_to_np(torch.float16) == np.dtype(np.float16)

    def test_float32(self):
        assert convert_dtype_to_np(torch.float32) == np.dtype(np.float32)

    def test_float64(self):
        assert convert_dtype_to_np(torch.float64) == np.dtype(np.float64)

    def test_bfloat16(self):
        assert convert_dtype_to_np(torch.bfloat16) == np.dtype(bfloat16)

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            convert_dtype_to_np(torch.int32)

    def test_unsupported_complex_raises(self):
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            convert_dtype_to_np(torch.complex64)


class TestTensorToNumpy:
    def test_float32_roundtrip(self):
        t = torch.tensor([1.0, 2.5, -3.0], dtype=torch.float32)
        arr = tensor_to_numpy(t)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, t.numpy())

    def test_float16_roundtrip(self):
        t = torch.tensor([1.0, 2.5, -3.0], dtype=torch.float16)
        arr = tensor_to_numpy(t)
        assert arr.dtype == np.float16
        np.testing.assert_array_equal(arr, t.numpy())

    def test_float64_roundtrip(self):
        t = torch.tensor([1.0, 2.5, -3.0], dtype=torch.float64)
        arr = tensor_to_numpy(t)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, t.numpy())

    def test_bfloat16_dtype(self):
        t = torch.tensor([1.0, 2.5, -3.0], dtype=torch.bfloat16)
        arr = tensor_to_numpy(t)
        assert arr.dtype == bfloat16

    def test_bfloat16_preserves_values(self):
        t = torch.tensor([1.0, -0.5, 3.14, 0.0, 100.0], dtype=torch.bfloat16)
        arr = tensor_to_numpy(t)
        expected = t.float().numpy().astype(bfloat16)
        np.testing.assert_array_equal(arr, expected)

    def test_bfloat16_preserves_bit_pattern(self):
        t = torch.tensor([1.0, -2.0, 0.001], dtype=torch.bfloat16)
        arr = tensor_to_numpy(t)
        torch_bits = t.view(torch.uint16).numpy()
        numpy_bits = arr.view(np.uint16)
        np.testing.assert_array_equal(torch_bits, numpy_bits)

    def test_bfloat16_special_values(self):
        t = torch.tensor([0.0, float("inf"), float("-inf")], dtype=torch.bfloat16)
        arr = tensor_to_numpy(t)
        assert arr[0] == bfloat16(0.0)
        assert np.isinf(arr[1])
        assert np.isinf(arr[2])

    def test_2d_tensor(self):
        t = torch.randn(3, 4, dtype=torch.float32)
        arr = tensor_to_numpy(t)
        assert arr.shape == (3, 4)
        np.testing.assert_array_equal(arr, t.numpy())

    def test_2d_bfloat16(self):
        t = torch.randn(3, 4, dtype=torch.bfloat16)
        arr = tensor_to_numpy(t)
        assert arr.shape == (3, 4)
        assert arr.dtype == bfloat16

    def test_empty_tensor(self):
        t = torch.tensor([], dtype=torch.float32)
        arr = tensor_to_numpy(t)
        assert len(arr) == 0

    def test_empty_bfloat16(self):
        t = torch.tensor([], dtype=torch.bfloat16)
        arr = tensor_to_numpy(t)
        assert len(arr) == 0
        assert arr.dtype == bfloat16
