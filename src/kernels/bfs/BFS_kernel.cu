__global__ void BFS_kernel_warp(
    unsigned int *levels,
    unsigned int *edgeArray,
    unsigned int *edgeArrayAux,
    int W_SZ,
    unsigned int numVertices,
    int curr,
    int *flag)
{
    int tid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    int CHUNK_SZ = CHUNK_SIZE;
    int W_OFF = tid % W_SZ;
    int W_ID = tid / W_SZ;
    int v1= W_ID * CHUNK_SZ;
    int chk_sz=CHUNK_SZ+1;

    if((v1+CHUNK_SZ)>=numVertices) {
        chk_sz =  numVertices-v1+1;
        if(chk_sz<0)
            chk_sz=0;
    }

    #if UNROLL_OUTER_LOOP
    #pragma unroll
    #endif
    for(int v=v1; v< chk_sz-1+v1; v++) {
        if(levels[v] == curr) {
            unsigned int num_nbr = edgeArray[v+1]-edgeArray[v];
            unsigned int nbr_off = edgeArray[v];
            
            #if UNROLL_INNER_LOOP
            #pragma unroll
            #endif
            for(int i=W_OFF; i<num_nbr; i+=W_SZ) {
                int v = edgeArrayAux[i + nbr_off];
                if(levels[v]==UINT_MAX) {
                        levels[v] = curr + 1;
                        *flag = 1;
                }
            }
        }
    }
}