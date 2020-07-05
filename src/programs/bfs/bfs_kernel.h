#ifndef BFS_KERNEL_H_
#define BFS_KERNEL_H_

#include<cuda.h>

#define get_queue_index(tid) ((tid%NUM_P_PER_MP))
#define get_queue_offset(tid) ((tid%NUM_P_PER_MP)*W_Q_SIZE)


__global__ void BFS_kernel_warp(
        unsigned int *levels,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        int W_SZ,
        unsigned int numVertices,
        int curr,
        int *flag);

#endif        //BFS_KERNEL_H
