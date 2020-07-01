#include "bfs_kernel.h"

// BFS depends on atomic instructions.  NVCC will generate errors if
// the code is compiled for CC < 1.2.  So, we use this macro and stubs
// so the code will compile cleanly.  If run on CC < 1.2, it will
// return a "NoResult" flag.
#if __CUDA_ARCH__ >= 120

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
__global__ void BFS_kernel_warp(
        unsigned int *levels,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        int W_SZ,
        unsigned int numVertices,
        int curr,
        int *flag)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int CHUNK_SZ = CHUNK_SIZE;
    int W_OFF = tid % W_SZ;
    int W_ID = tid / W_SZ;
    int v1= W_ID * CHUNK_SZ;
    int chk_sz=CHUNK_SZ+1;

    if((v1+CHUNK_SZ)>=numVertices)
    {
        chk_sz =  numVertices-v1+1;
        if(chk_sz<0)
            chk_sz=0;
    }

    #if UNROLL_OUTER_LOOP
    #pragma unroll
    #endif
    for(int v=v1; v< chk_sz-1+v1; v++)
    {
        if(levels[v] == curr)
        {
            unsigned int num_nbr = edgeArray[v+1]-edgeArray[v];
            unsigned int nbr_off = edgeArray[v];
            
            #if UNROLL_INNER_LOOP
            #pragma unroll
            #endif
            for(int i=W_OFF; i<num_nbr; i+=W_SZ)
            {
               int v = edgeArrayAux[i + nbr_off];
               if(levels[v]==UINT_MAX)
               {
                    levels[v] = curr + 1;
                    *flag = 1;
               }
            }
        }
    }
}



#else
// No atomics are available, compile with stubs.
__global__ void BFS_kernel_warp(
        unsigned int *levels,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        int W_SZ,
        int CHUNK_SZ,
        unsigned int numVertices,
        int curr,
        int *flag) { ; }
#endif
