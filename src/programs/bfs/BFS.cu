#include <cuda.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string.h>

#include "bfs_kernel.h"
#include "cudacommon.h"
#include "Graph.h"
#include "OptionParser.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
//TODO: Check if hostfile option in driver file adds automatically
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("graph_file", OPT_STRING, "random", "name of graph file");
    op.addOption("degree", OPT_INT, "2", "average degree of nodes");
    op.addOption("algo", OPT_INT, "1", "1-IIIT BFS 2-UIUC BFS ");
    op.addOption("dump-pl", OPT_BOOL, "false",
            "enable dump of path lengths to file");
    op.addOption("source_vertex", OPT_INT, "0",
            "vertex to start the traversal from");
    op.addOption("global-barrier", OPT_BOOL, "false",
            "enable the use of global barrier in UIUC BFS");
}


// ****************************************************************************
// Function: verify_results
//
// Purpose:
//  Verify BFS results by comparing the output path lengths from cpu and gpu
//  traversals
//
// Arguments:
//   cpu_cost: path lengths calculated on cpu
//   gpu_cost: path lengths calculated on gpu
//   numVerts: number of vertices in the given graph
//   out_path_lengths: specify if path lengths should be dumped to files
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
unsigned int verify_results(unsigned int *cpu_cost, unsigned int *gpu_cost,
                            unsigned int numVerts,  bool out_path_lengths)
{
    unsigned int unmatched_nodes=0;
    for(int i=0;i<numVerts;i++)
    {
        if(gpu_cost[i]!=cpu_cost[i])
        {
            unmatched_nodes++;
        }
    }

    // If user wants to write path lengths to file
    if(out_path_lengths)
    {
        std::ofstream bfs_out_cpu("bfs_out_cpu.txt");
        std::ofstream bfs_out_gpu("bfs_out_cuda.txt");
        for(int i=0;i<numVerts;i++)
        {
            if(cpu_cost[i]!=UINT_MAX)
                bfs_out_cpu<<cpu_cost[i]<<"\n";
            else
                bfs_out_cpu<<"0\n";

            if(gpu_cost[i]!=UINT_MAX)
                bfs_out_gpu<<gpu_cost[i]<<"\n";
            else
                bfs_out_gpu<<"0\n";
        }
        bfs_out_cpu.close();
        bfs_out_gpu.close();
    }
    return unmatched_nodes;
}

// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Runs the BFS benchmark using method 1 (IIIT-BFS method)
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//   G: input graph
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunTest(OptionParser &op, Graph *G)
{
    typedef char frontier_type;
    typedef unsigned int cost_type;

    // Get graph info
    unsigned int *edgeArray=G->GetEdgeOffsets();
    unsigned int *edgeArrayAux=G->GetEdgeList();
    unsigned int adj_list_length=G->GetAdjacencyListLength();
    unsigned int numVerts = G->GetNumVertices();
    unsigned int numEdges = G->GetNumEdges();

    int *flag;

    // Allocate pinned memory for frontier and cost arrays on CPU
    cost_type  *costArray;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&costArray,
                                  sizeof(cost_type)*(numVerts)));
    CUDA_SAFE_CALL(cudaMallocHost((void **)&flag,
                                  sizeof(int)));

    // Variables for GPU memory
    // Adjacency lists
    unsigned int *d_edgeArray=NULL,*d_edgeArrayAux=NULL;
    // Cost array
    cost_type  *d_costArray;
    // Flag to check when traversal is complete
    int *d_flag;

    // Allocate memory on GPU
    CUDA_SAFE_CALL(cudaMalloc(&d_costArray,sizeof(cost_type)*numVerts));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeArray,sizeof(unsigned int)*(numVerts+1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeArrayAux,
                                        adj_list_length*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_flag,sizeof(int)));

    // Initialize frontier and cost arrays
    for (int index = 0; index < numVerts; index++) {
        costArray[index]=UINT_MAX;
    }

    // Set vertex to start traversal from
    const unsigned int source_vertex=op.getOptionInt("source_vertex");
    costArray[source_vertex]=0;

    // Transfer frontier, cost array and adjacency lists on GPU
    CUDA_SAFE_CALL(cudaMemcpy(d_costArray, costArray,
                   sizeof(cost_type)*numVerts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeArray, edgeArray,
                   sizeof(unsigned int)*(numVerts+1),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeArrayAux,edgeArrayAux,
                 sizeof(unsigned int)*adj_list_length,cudaMemcpyHostToDevice));
    
    cudaTextureObject_t textureObjEA=0;
    #if TEXTURE_MEMORY_EA1 == 2
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = d_edgeArray;
        resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        resDesc.res.linear.desc.x = 32;
        resDesc.res.linear.sizeInBytes = sizeof(unsigned int)*(numVerts+1);
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.addressMode = cudaAddressModeWrap;
        CUDA_SAFE_CALL(cudaCreateTextureObject(&textureObjEA, &resDesc, &texDesc, NULL));
    #elif TEXTURE_MEMORY_EA1 == 1
        //Bind a 1D texture to the edgeArray array
        CUDA_SAFE_CALL(cudaBindTexture(0, textureRefEA, d_edgeArray, sizeof(unsigned int)*(numVerts+1)));
    #endif
    
    cudaTextureObject_t textureObjEAA=0;
    #if TEXTURE_MEMORY_EAA == 2
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = d_edgeArrayAux;
        resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        resDesc.res.linear.desc.x = 32;
        resDesc.res.linear.sizeInBytes = adj_list_length*sizeof(unsigned int);
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;        
        CUDA_SAFE_CALL(cudaCreateTextureObject(&textureObjEAA, &resDesc, &texDesc, NULL));
    #elif TEXTURE_MEMORY_EAA == 1
        // Bind a 1D texture to the position array
        CUDA_SAFE_CALL(cudaBindTexture(0, textureRefEAA, d_edgeArrayAux, adj_list_length*sizeof(unsigned int)));
    #endif
            
    // Get the device properties for kernel configuration
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,device);

    // Set the kernel configuration
    int chunkFactor = CHUNK_SIZE/32;
    int numThreads = BLOCK_SIZE;
    int numBlocks = 0;
    numBlocks = (int)ceil((double)numVerts/(double)numThreads/(double)chunkFactor);
    
    if (numBlocks > devProp.maxGridSize[0]) {
        std::cout << "Max number of blocks exceeded";
        return;
    }

    unsigned int *cpu_cost = new unsigned int[numVerts];
    // Perform cpu bfs traversal for verifying results
    G->GetVertexLengths(cpu_cost,source_vertex);

    // Start the benchmark
    int passes = op.getOptionInt("passes");

    for (int j = 0; j < passes; j++) {
        // Flag set when there are nodes to be traversed in frontier
        *flag = 1;

        int iters = 0;
        int W_SZ = 32;
        // While there are nodes to traverse
        while (*flag) {
            // Set flag to 0
            *flag=0;
            CUDA_SAFE_CALL(cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice));

            BFS_kernel_warp<<<numBlocks,numThreads>>>
                (d_costArray, d_edgeArray, textureObjEA, d_edgeArrayAux, textureObjEAA, W_SZ, numVerts, iters, d_flag);
            CHECK_CUDA_ERROR();

            // Read flag
            CUDA_SAFE_CALL(cudaMemcpy(flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
            iters++;
        }


        // Copy the cost array from GPU to CPU
        CUDA_SAFE_CALL(cudaMemcpy(costArray,d_costArray,
                       sizeof(cost_type)*numVerts,cudaMemcpyDeviceToHost));

        // Count number of vertices visited
        unsigned int numVisited = 0;
        for (int i = 0; i < numVerts; i++) {
            if (costArray[i]!=UINT_MAX)
                numVisited++;
        }

        bool dump_paths = op.getOptionBool("dump-pl");
        // Verify Results against serial BFS
        unsigned int unmatched_verts=verify_results(cpu_cost,costArray,numVerts,dump_paths);

        if (unmatched_verts!=0) {
            std::cout << "Failed\n";
            cerr << "Error: incorrect computed result." << endl;
            return;
        }

        if (j==passes-1) //if passes completed break;
            break;

        // Reset the arrays to perform BFS again
        for (int index=0;index<numVerts;index++)
        {
            costArray[index]=UINT_MAX;
        }
        costArray[source_vertex]=0;

        CUDA_SAFE_CALL(cudaMemcpy(d_costArray, costArray,
                       sizeof(cost_type)*numVerts, cudaMemcpyHostToDevice));

    }

    // Clean up
    delete[] cpu_cost;
    CUDA_SAFE_CALL(cudaFreeHost(costArray));
    #if TEXTURE_MEMORY_EA1 == 2
    CUDA_SAFE_CALL(cudaDestroyTextureObject(textureObjEA));
    #elif TEXTURE_MEMORY_EA1 == 1
    CUDA_SAFE_CALL(cudaUnbindTexture(textureRefEA));
    #endif
    #if TEXTURE_MEMORY_EAA == 2
    CUDA_SAFE_CALL(cudaDestroyTextureObject(textureObjEAA));
    #elif TEXTURE_MEMORY_EAA == 1
    CUDA_SAFE_CALL(cudaUnbindTexture(textureRefEA));
    #endif
    CUDA_SAFE_CALL(cudaFree(d_costArray));
    CUDA_SAFE_CALL(cudaFree(d_edgeArray));
    CUDA_SAFE_CALL(cudaFree(d_edgeArrayAux));
}


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the BFS benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op)
{

    // First, check if the device supports atomics, which are required
    // for this benchmark.  If not, return the "NoResult" sentinel.int device;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    //adjacency list variables
    //number of vertices and edges in graph
    unsigned int numVerts,numEdges;
    //Get the graph filename
    string inFileName = op.getOptionString("graph_file");
    //max degree in graph
    Graph *G=new Graph();

    unsigned int **edge_ptr1 = G->GetEdgeOffsetsPtr();
    unsigned int **edge_ptr2 = G->GetEdgeListPtr();
    //Load simple k-way tree or from a file
    if (inFileName == "random")
    {
        //Load simple k-way tree
        unsigned int prob_sizes[5] = {1000,10000,100000,1000000,10000000};
        numVerts = prob_sizes[op.getOptionInt("size")-1];
        int avg_degree = op.getOptionInt("degree");
        if(avg_degree<1)
            avg_degree=1;

        //allocate memory for adjacency lists
        //edgeArray =new unsigned int[numVerts+1];
        //edgeArrayAux=new unsigned int[numVerts*(avg_degree+1)];

        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr1,
                        sizeof(unsigned int)*(numVerts+1)));
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr2,
                        sizeof(unsigned int)*(numVerts*(avg_degree+1))));

        //Generate simple tree
        G->GenerateSimpleKWayGraph(numVerts,avg_degree);
    }
    else
    {
        //open the graph file
        FILE *fp=fopen(inFileName.c_str(),"r");
        if(fp==NULL)
        {
            std::cerr <<"Error: Graph Input File Not Found." << endl;
            return;
        }

        //get the number of vertices and edges from the first line
        const char delimiters[]=" \n";
        char charBuf[MAX_LINE_LENGTH];
        fgets(charBuf,MAX_LINE_LENGTH,fp);
        char *temp_token = strtok (charBuf, delimiters);
        while(temp_token[0]=='%')
        {
            fgets(charBuf,MAX_LINE_LENGTH,fp);
            temp_token = strtok (charBuf, delimiters);
        }
        numVerts=atoi(temp_token);
        temp_token = strtok (NULL, delimiters);
        numEdges=atoi(temp_token);

        //allocate pinned memory
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr1,
                        sizeof(unsigned int)*(numVerts+1)));
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr2,
                        sizeof(unsigned int)*(numEdges*2)));

        fclose(fp);
        //Load the specified graph
        G->LoadMetisGraph(inFileName.c_str());
    }

    RunTest(op,G);
    
    //Clean up
    delete G;
    CUDA_SAFE_CALL(cudaFreeHost(*edge_ptr1));
    CUDA_SAFE_CALL(cudaFreeHost(*edge_ptr2));
}
