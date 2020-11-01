## Parameters and Search Spaces
The algorithms that has constrainsts for parameters, includes it in the related section.

### Triad
| Parameter | Description | Search Space |
| --- | --- | --- |
| `BLOCK_SIZE` | Block size to launch the `triad` kernel with. | {1, ..., `Max GPU threads per block`} |
| `WORK_PER_THREAD` | The amount of work performed by each GPU thread. | {1, ..., 10} |
| `LOOP_UNROLL_TRIAD` | Whether to unroll the loop in the `triad` function or not. | {False (0), True (1)} |
| `PRECISION` | Whether to use single-precision floating-point or double-precision floating-point for the computations. | {`float` (32), `double` (64)} |

### BFS

### Molecular Dynamics  (MD)
| Parameter | Description | Search Space |
| --- | --- | --- |
| `BLOCK_SIZE` | Block size to launch the `compute_lj_force` kernel with. | {1, ..., `Max GPU threads per block`} |
| `PRECISION` | Whether to use single-precision floating-point or double-precision floating-point for the computations. | {`float` (32), `double` (64)} |
| `TEXTURE_MEMORY` | Whether to use texture memory or not for the input data to the `compute_lj_force` kernel. | {False (0), True (1)} |
| `WORK_PER_THREAD` | The amount of work performed by each GPU thread. | {1, ..., 5} |

### SPMV

### Sort
| Parameter | Description | Search Space |
| --- | --- | --- |
| `LOOP_UNROLL_LSB` |  Whether to unroll the loop in the `scanLSB` function or not. | {False (0), True (1)} |
| `LOOP_UNROLL_LOCAL_MEMORY` |  Whether to unroll the loop in the `scanLocalMem` function or not. | {False (0), True (1)} |
| `SCAN_DATA_SIZE` | Size of the data type that is used for the `scan` functions. It chooses either `uint2`, `uint4` or a custom `uint8` data type. | {2, 4, 8} |
| `SORT_DATA_SIZE` | Size of the data type that is used for the `sort` functions. It chooses either `uint2`, `uint4` or a custom `uint8` data type. | {2, 4, 8} |
| `SCAN_BLOCK_SIZE` | Block size to launch the `scan` kernels with. | {16, 32, 64, 128, 256, 512, 1024} |
| `SORT_BLOCK_SIZE` | Block size to launch the `sort` kernels with. | {16, 32, 64, 128, 256, 512, 1024} |
| `INLINE_LSB` | Whether to inline the `scanLSB` function or not. | {False (0), True (1)} |
| `INLINE_SCAN` | Whether to inline the `scan4` function or not. | {False (0), True (1)} |
| `INLINE_LOCAL_MEMORY` | Whether to inline the `scanLocalMem` function or not. | {False (0), True (1)} |

#### Constraints
| Constraint | Description |
| --- | --- |
| `SCAN_DATA_SIZE * SCAN_BLOCK_SIZE = SORT_DATA_SIZE * SORT_BLOCK_SIZE` | The ratio between sort and scan data- and block sizes needs to be equal. |
| `(8 * SCAN_DATA_SIZE * SCAN_BLOCK_SIZE + 128) ≤ available_shared_memory` | Available shared memory for the selected GPU can not be less than needed shared memory in the `reorderData` kernel. |

### MD5 Hash

### Reduction
| Parameter | Description | Search Space |
| --- | --- | --- |
| `BLOCK_SIZE` | Block size to launch the `reduce` kernel with. | 2^i where i is in the range [0, 11] with a max of threads per block and excluding 32 |
| `GRID_SIZE` | Grid size to launch the `reduce` kernel with. | {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} |
| `PRECISION` | Whether to use single-precision floating-point or double-precision floating-point for the computations. | {`float` (32), `double` (64)} |
| `COMPILER_OPTIMIZATION_HOST` | The level of optimization that are applied by the compiler for the host code. | {1, ..., 4} |
| `COMPILER_OPTIMIZATION_DEVICE` | The level of optimization that are applied by the compiler for the device code. | {1, ..., 4} |
| `USE_FAST_MATH` | Whether or not to use the compiler option to use the fast math library. | {False (0), True (1)} |
| `MAX_REGISTERS` | Maximum number of registers that GPU functions can use. | {-1, 20, 40, 60, 80, 100, 120} |
| `GPUS` | Number of GPUs to perform the computation of `redcution` on. | {1, ..., `Connected GPUs`} |
| `LOOP_UNROLL_REDUCE_1` | Whether to unroll the first loop in the `reduce` function or not. | {False (0), True (1)} |
| `LOOP_UNROLL_REDUCE_2` | Whether to unroll the second loop in the `reduce` function or not. | {False (0), True (1)} |
| `TEXTURE_MEMORY` | Whether to use texture memory or not for the input data to the `reduce` kernel. | {False (0), True (1)} |

### Scan

### Stencil 2D
| Parameter | Description | Search Space |
| --- | --- | --- |
| `GPUS` | Number of GPUs to perform the computation of `StencilKernel` on. | {1, ..., `Connected GPUs`} |
