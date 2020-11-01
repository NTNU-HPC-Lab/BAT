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
| Parameter | Description | Search Space |
| --- | --- | --- |
| `BLOCK_SIZE` | The block size used for launching the kernel. | {1, ..., `Max GPU threads per block` or 1000 if size=1} |
| `CHUNK_FACTOR` | Factor multiplied with a chunk size which is the work per thread. | {1, 2, 4, 8} |
| `TEXTURE_MEMORY_EA1` | Not use texture memory, use texture reference (older version) or use texture object (newer version). | {0, 1, 2} |
| `TEXTURE_MEMORY_EAA` | Not use texture memory, use texture reference (older version) or use texture object (newer version). | {0, 1, 2} |

### Molecular Dynamics  (MD)
| Parameter | Description | Search Space |
| --- | --- | --- |
| `BLOCK_SIZE` | Block size to launch the `compute_lj_force` kernel with. | {1, ..., `Max GPU threads per block`} |
| `PRECISION` | Whether to use single-precision floating-point or double-precision floating-point for the computations. | {`float` (32), `double` (64)} |
| `TEXTURE_MEMORY` | Whether to use texture memory or not for the input data to the `compute_lj_force` kernel. | {False (0), True (1)} |
| `WORK_PER_THREAD` | The amount of work performed by each GPU thread. | {1, ..., 5} |

### SPMV
| Parameter | Description | Search Space |
| --- | --- | --- |
| `BLOCK_SIZE` | The block size used for the kernel. | {1, ..., `Max GPU threads per block`} |
| `PRECISION` | Use single or double precision on variables. | {32, 64} |
| `FORMAT` | Which format to use for SpMV. <br> 0: ELLPACK-R <br>1: CSR Normal Scalar <br>2: CSR Padded Scalar <br>3: CSR Normal Vector <br>4: CSR Padded Vector | {0, 1, 2, 3, 4} |
| `UNROLL_LOOP_2` | Not unroll or unroll loop. | {0, 1} |
| `TEXTURE_MEMORY` | Not use texture memory or use the newer version of texture memory (texture objects). | {0, 1} |

#### Constraints
| Constraint | Description |
| --- | --- |
| `FORMAT < 3 or BLOCK_SIZE % 32 == 0` | Format 3 or 4, which is the CSR Vector formats, needs to have block size that is divisible by 32. |
| `FORMAT > 2 or UNROLL_LOOP_2 < 1` | The loop exists only in the CSR vector kernel, which means that the loop unroll parameter should be 0 when the format is something else than CSR vector. |


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
| Parameter | Description | Search Space |
| --- | --- | --- |
| `BLOCK_SIZE` | The block size used for the kernel. | {1, ..., `Max GPU threads per block`} |
| `ROUND_STYLE` | Which MD5 round style to use. | {0, 1} |
| `UNROLL_LOOP_1` | Not unroll or unroll loop 1. | {0, 1} |
| `UNROLL_LOOP_2` | Not unroll or unroll loop 2. | {0, 1} |
| `UNROLL_LOOP_3` | Not unroll or unroll loop 3. | {0, 1} |
| `INLINE_1` | Not inline or inline kernel 1. | {0, 1} |
| `INLINE_2` | Not inline or inline kernel 2. | {0, 1} |
| `WORK_PER_THREAD_FACTOR` | Factor for setting how much work a thread should do. | {1, 2, 3, 4, 5} |

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
| Parameter | Description | Search Space |
| --- | --- | --- |
| `BLOCK_SIZE` | The block size used for the kernels. | {16, 64, 128, 256, 512} |
| `GRID_SIZE` | The grid size used for the kernels. | {1, 2, 4, 8, 16, 32, 64, 128, 256, 512} |
| `PRECISION` | Use single or double precision. | {32, 64} |
| `UNROLL_LOOP_1` | Not unroll or unroll loop 1. | {0, 1} |
| `UNROLL_LOOP_2` | Not unroll or unroll loop 2. | {0, 1} |
| `USE_FAST_MATH` | If fast math will be used when compiling. | {0, 1} |
| `OPTIMIZATION_LEVEL_HOST` | Compiler optimization level for host code. | {0, 1, 2, 3} |
| `OPTIMIZATION_LEVEL_DEVICE` | Compiler optimization level for GPU code. | {0, 1, 2, 3} |
| `MAX_REGISTERS` | The amount of max registers to be used for the GPU kernels. -1 means to not set this compiler option. | {-1, 20, 40, 60, 80, 100, 120} |
| `GPUS` | The number of GPUs used in the computations.  | {1, ..., `Connected GPUs`} |

#### Constraints
| Constraint | Description |
| --- | --- |
| `GRID_SIZE <= BLOCK_SIZE` | Grid size needs to be less than or equal to block size. |


### Stencil 2D
| Parameter | Description | Search Space |
| --- | --- | --- |
| `GPUS` | Number of GPUs to perform the computation of `StencilKernel` on. | {1, ..., `Connected GPUs`} |
