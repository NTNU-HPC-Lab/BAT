#ifndef BFS_KERNEL_H_
#define BFS_KERNEL_H_

#include<cuda.h>

#define get_queue_index(tid) ((tid%NUM_P_PER_MP))
#define get_queue_offset(tid) ((tid%NUM_P_PER_MP)*W_Q_SIZE)

texture<unsigned int, 1, cudaReadModeElementType> textureRefEA;
texture<unsigned int, 1, cudaReadModeElementType> textureRefEAA;

//Sungpack Hong, Sang Kyun Kim, Tayo Oguntebi, and Kunle Olukotun. 2011.
//Accelerating CUDA graph algorithms at maximum warp.
//In Proceedings of the 16th ACM symposium on Principles and practice of
//parallel programming (PPoPP '11). ACM, New York, NY, USA, 267-276.
// ****************************************************************************
// Function: BFS_kernel_warp
//
// Purpose:
//   Perform BFS on the given graph
//
// Arguments:
//   levels: array that stores the level of vertices
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   W_SZ: the warp size to use to process vertices
//   CHUNK_SZ: the number of vertices each warp processes
//   numVertices: number of vertices in the given graph
//   curr: the current BFS level
//   flag: set when more vertices remain to be traversed
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
//template <typename texReaderEA, typename texReaderEAA>
extern "C" __global__ void BFS_kernel_warp(
    unsigned int *levels,
    unsigned int *edgeArray,
    cudaTextureObject_t textureObjEA,
    unsigned int *edgeArrayAux,
    cudaTextureObject_t textureObjEAA,
    int W_SZ,
    unsigned int numVertices,
    int curr,
    int *flag)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int CHUNK_SZ = CHUNK_FACTOR*32;
    int W_OFF = tid % W_SZ;
    int W_ID = tid / W_SZ;
    int v1= W_ID * CHUNK_SZ;
    int chk_sz=CHUNK_SZ+1;

    if((v1+CHUNK_SZ)>=numVertices) {
        chk_sz =  numVertices-v1+1;
        if(chk_sz<0) {
            chk_sz=0;
        }
    }

    for (int v=v1; v< chk_sz-1+v1; v++) {
        if (levels[v] == curr) {
            #if TEXTURE_MEMORY_EA1 == 2
                unsigned int num_nbr = tex1Dfetch<unsigned int>(textureObjEA, v+1) - tex1Dfetch<unsigned int>(textureObjEA, v);
                unsigned int nbr_off = tex1Dfetch<unsigned int>(textureObjEA, v);
            #elif TEXTURE_MEMORY_EA1 == 1
                unsigned int num_nbr = tex1Dfetch(textureRefEA, v+1) - tex1Dfetch(textureRefEA, v);
                unsigned int nbr_off = tex1Dfetch(textureRefEA, v);
            #else
                unsigned int num_nbr = edgeArray[v+1]-edgeArray[v];
                unsigned int nbr_off = edgeArray[v];
            #endif
            
            for (int i=W_OFF; i<num_nbr; i+=W_SZ) {
                #if TEXTURE_MEMORY_EAA == 2
                    int v = tex1Dfetch<unsigned int>(textureObjEAA, (i + nbr_off));
                #elif TEXTURE_MEMORY_EAA == 1
                    int v = tex1Dfetch(textureRefEAA, (i + nbr_off));
                #else
                    int v = edgeArrayAux[i + nbr_off];
                #endif
                if (levels[v]==UINT_MAX) {
                    levels[v] = curr + 1;
                    *flag = 1;
                }
            }
        }
    }
}

#endif        //BFS_KERNEL_H
