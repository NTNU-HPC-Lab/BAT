import pytest
import numpy as np
import cupy as cp

from batbench.backends.cuda_kernel_runner.arg_handler import ArgHandler, UnsupportedMemoryTypeError

pytestmark = pytest.mark.backends

def test_arg_handler_scalar_type_conversion():
    arg_handler = ArgHandler({
        "KernelSpecification": {
            "KernelName": "GEMM",
            "Arguments": [
                {
                    "Name": "x",
                    "MemoryType": "Scalar",
                    "Type": "int32",
                    "FillValue": 5
                }
            ]
        }
    })

    expected_arg_value = np.int32(5)
    args, _ = arg_handler.populate_args().get_function_args()

    assert args[0] == expected_arg_value

def test_arg_handler_vector_type_conversion():
    arg_handler = ArgHandler({
        "KernelSpecification": {
            "KernelName": "GEMM",
            "Arguments": [
                {
                    "Name": "x",
                    "MemoryType": "Vector",
                    "Type": "int32",
                    "FillType": "Constant",
                    "FillValue": 7,
                    "Size": 3
                }
            ]
        }
    })

    expected_arg_value = cp.array([7, 7, 7], dtype=np.int32)
    args, _ = arg_handler.populate_args().get_function_args()

    cp.testing.assert_array_equal(args[0], expected_arg_value)

def test_arg_handler_custom_type():
    arg_handler = ArgHandler({
        "KernelSpecification": {
            "KernelName": "GEMM",
            "Arguments": [
                {
                    "Name": "x",
                    "MemoryType": "Scalar",
                    "Type": "float2",
                    "FillValue": [1.0, 2.0]
                }
            ]
        }
    })

    expected_arg_value = np.array([(1.0, 2.0)], dtype=[('x', 'f4'), ('y', 'f4')])
    args, _ = arg_handler.populate_args().get_function_args()

    assert np.array_equal(args[0], expected_arg_value)

def test_unsupported_memory_type_error():
    arg_handler = ArgHandler({
        "KernelSpecification": {
            "KernelName": "GEMM",
            "Arguments": [
                {
                    "Name": "x",
                    "MemoryType": "UnsupportedType",
                    "Type": "int32",
                    "FillValue": 5
                }
            ]
        }
    })

    with pytest.raises(UnsupportedMemoryTypeError):
        arg_handler.populate_args()

def test_constant_memory_args():
    arg_handler = ArgHandler({
        "KernelSpecification": {
            "KernelName": "GEMM",
            "Arguments": [
                {
                    "Name": "x",
                    "MemoryType": "Vector",
                    "MemType": "Constant",
                    "Type": "int32",
                    "FillType": "Constant",
                    "FillValue": 7,
                    "Size": 3
                }
            ]
        }
    })

    expected_cmem_arg_value = cp.array([7, 7, 7], dtype=np.int32)
    _, cmem_args = arg_handler.populate_args().get_function_args()

    cp.testing.assert_array_equal(cmem_args["x"], expected_cmem_arg_value)

def test_unsupported_fill_type_error():
    arg_handler = ArgHandler({
        "KernelSpecification": {
            "KernelName": "GEMM",
            "Arguments": [
                {
                    "Name": "x",
                    "MemoryType": "Vector",
                    "Type": "int32",
                    "FillType": "UnsupportedFillType",
                    "Size": 3
                }
            ]
        }
    })

    with pytest.raises(ValueError):
        arg_handler.populate_args()
