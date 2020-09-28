#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <list>
#include "cudacommon.h"

extern "C" {

#include "bfs_kernel.cu"

using namespace std;

#define MAX_LINE_LENGTH 500000

class Graph
{
    unsigned int num_verts;
    unsigned int num_edges;
    unsigned int adj_list_length;
    unsigned int *edge_offsets;
    unsigned int *edge_list;
    unsigned int *edge_costs;
    unsigned int max_degree;
    int graph_type;

    bool if_delete_arrays;

    void SetAllCosts(unsigned int c);
    public:
    Graph();
    ~Graph();
    void LoadMetisGraph(const char *filename);
    void SaveMetisGraph(const char *filename);
    unsigned int GetNumVertices();
    unsigned int GetNumEdges();
    unsigned int GetMaxDegree();

    unsigned int *GetEdgeOffsets();
    unsigned int *GetEdgeList();
    unsigned int *GetEdgeCosts();

    unsigned int **GetEdgeOffsetsPtr();
    unsigned int **GetEdgeListPtr();
    unsigned int **GetEdgeCostsPtr();

    unsigned int *GetVertexLengths(unsigned int *cost,unsigned int source);
    int GetMetisGraphType();
    unsigned int GetAdjacencyListLength();
    void GenerateSimpleKWayGraph(unsigned int verts,unsigned int degree);
};

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
float RunTest(Graph *G)
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
    const unsigned int source_vertex=0;
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
    int numThreads = BLOCK_SIZE;
    int numBlocks = (int)ceil((double)numVerts/(double)numThreads/(double)CHUNK_FACTOR);
    
    if (numBlocks > devProp.maxGridSize[0]) {
        std::cout << "Max number of blocks exceeded";
        throw "Max number of blocks exceeded";
        return 0.0;
    }

    unsigned int *cpu_cost = new unsigned int[numVerts];
    // Perform cpu bfs traversal for verifying results
    G->GetVertexLengths(cpu_cost,source_vertex);

    // Start the benchmark
    int passes = 10;

    // Initialize timers
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    float totalElapsedTime = 0.0;

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

            // Start the timing
            cudaEventRecord(start, 0);

            BFS_kernel_warp<<<numBlocks,numThreads>>>
                (d_costArray, d_edgeArray, textureObjEA, d_edgeArrayAux, textureObjEAA, W_SZ, numVerts, iters, d_flag);
            CHECK_CUDA_ERROR();

            // Stop the events and save elapsed time
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            totalElapsedTime += elapsedTime;

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

        bool dump_paths = false;
        // Verify Results against serial BFS
        unsigned int unmatched_verts=verify_results(cpu_cost,costArray,numVerts,dump_paths);

        if (unmatched_verts!=0) {
            std::cout << "Failed\n";
            cerr << "Error: incorrect computed result." << endl;
            throw "Error: incorrect computed result.";
            return 0.0;
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
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    
    return totalElapsedTime;
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
float RunBenchmark()
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

    //max degree in graph
    Graph *G=new Graph();

    unsigned int **edge_ptr1 = G->GetEdgeOffsetsPtr();
    unsigned int **edge_ptr2 = G->GetEdgeListPtr();

    //Load simple k-way tree
    unsigned int prob_sizes[5] = {1000,10000,100000,1000000,10000000};
    numVerts = prob_sizes[PROBLEM_SIZE - 1];
    int avg_degree = 2;
    if(avg_degree<1)
        avg_degree=1;

    CUDA_SAFE_CALL(cudaMallocHost(edge_ptr1,
                    sizeof(unsigned int)*(numVerts+1)));
    CUDA_SAFE_CALL(cudaMallocHost(edge_ptr2,
                    sizeof(unsigned int)*(numVerts*(avg_degree+1))));

    //Generate simple tree
    G->GenerateSimpleKWayGraph(numVerts,avg_degree);

    float totalTime = RunTest(G);
    
    //Clean up
    delete G;
    CUDA_SAFE_CALL(cudaFreeHost(*edge_ptr1));
    CUDA_SAFE_CALL(cudaFreeHost(*edge_ptr2));
    return totalTime;
} 



// Graph.cpp 

Graph::Graph()
{
    num_verts=0;
    num_edges=0;
    max_degree=0;
    adj_list_length=0;
    edge_offsets=NULL;
    edge_list=NULL;
    edge_costs=NULL;
    graph_type=-1;
    if_delete_arrays=false;
}

Graph::~Graph()
{
    if(if_delete_arrays)
    {
        delete[] edge_offsets;
        delete[] edge_list;
        if(graph_type==1)
            delete[] edge_costs;
    }
}


// ****************************************************************************
//  Method:  Graph::LoadMetisGraph
//
//  Purpose:
//      Loads a graph from METIS file format.
//
//  Arguments:
//    filename: file name of the graph to load
//
//  Programmer:  Aditya Sarwade
//  Creation:    June 16, 2011
//
//  Modifications:
//
// ****************************************************************************
void Graph::LoadMetisGraph(const char *filename)
{

	FILE *fp=fopen(filename,"r");
    assert(fp);
	char charBuf[MAX_LINE_LENGTH];
	const char delimiters[]=" \n";
	char *temp_token=NULL;

    while(1)
    {
        fgets(charBuf,MAX_LINE_LENGTH,fp);
        temp_token = strtok (charBuf, delimiters);

        if(temp_token==NULL)
            continue;

        else if(temp_token[0]=='%')
            continue;

        else
            break;

    }

    assert(isdigit(temp_token[0]));
	num_verts  = atoi(temp_token);
	temp_token = strtok (NULL, delimiters);
    assert(isdigit(temp_token[0]));
	num_edges=atoi(temp_token);
	temp_token = strtok (NULL, delimiters);
    if(temp_token==NULL)
    {
        graph_type = 0;
    }
    else
    {
        assert(isdigit(temp_token[0]));
        graph_type=atoi(temp_token);
        if(graph_type!=0 && graph_type!=1 && graph_type!=100)
        {
            std::cout<<"\nSupported metis graph types are 0 and 1";
            return;
        }
    }

    if(edge_offsets==NULL)
    {
        if_delete_arrays=true;
        edge_offsets=new unsigned int[num_verts+1];
        edge_list=new unsigned int[num_edges*2];
        if(graph_type == 1)
            edge_costs=new unsigned int[num_edges*2];
    }

    unsigned int cost=0;
	unsigned int offset=0;
	for(int index=1;index<=num_verts;index++)
	{
		fgets(charBuf,MAX_LINE_LENGTH,fp);

		temp_token=strtok(charBuf,delimiters);

		if(temp_token==NULL)
		{
			edge_offsets[index-1]=offset;
			continue;
		}
        if(temp_token[0]=='%')
        {
            continue;
        }
        assert(isdigit(temp_token[0]));

		unsigned int vert=atoi(temp_token);
		edge_offsets[index-1]=offset;
		edge_list[offset]=vert-1;

        if(graph_type==1)
		{
            temp_token=strtok(NULL,delimiters);
            assert(temp_token);
            assert(isdigit(temp_token[0]));
		    cost=atoi(temp_token);
            edge_costs[offset]=cost;
        }

		//temp_value=(index-1)*(num_verts)+(vert-1);
		offset++;
		while((temp_token=(strtok(NULL,delimiters))))
		{
            assert(isdigit(temp_token[0]));
			vert=atoi(temp_token);
			//temp_value=(index-1)*(num_verts)+(vert-1);
			edge_list[offset]=vert-1;

            if(graph_type==1)
            {
                temp_token=strtok(NULL,delimiters);
                assert(temp_token);
                assert(isdigit(temp_token[0]));
                cost=atoi(temp_token);
                edge_costs[offset]=cost;
            }

			offset++;
		}
        if(max_degree < offset-edge_offsets[index-1])
            max_degree=offset-edge_offsets[index-1];
	}

    adj_list_length=offset;

	//Add length of the adjacency list to last position
	edge_offsets[num_verts]=offset;
    adj_list_length=offset;
    fclose(fp);
}

// ****************************************************************************
//  Method:  Graph::SaveMetisGraph
//
//  Purpose:
//      Saves the graph in METIS file format.
//
//  Arguments:
//    filename: path to save the graph
//
//  Programmer:  Aditya Sarwade
//  Creation:    June 16, 2011
//
//  Modifications:
//
// ****************************************************************************
void Graph::SaveMetisGraph(const char *filename)
{
    FILE *fp=fopen(filename,"w");
    assert(fp);

    fprintf(fp,"%u %u",num_verts,num_edges);
    if(graph_type!=0)
        fprintf(fp," %d",graph_type);
    fprintf(fp,"\n");

    for(int i=0;i<num_verts;i++)
    {
        unsigned int offset=edge_offsets[i];
        unsigned int next  =edge_offsets[i+1];
        while(offset<next)
        {
            fprintf(fp,"%u ",edge_list[offset]+1);
            if(graph_type==1)
            {
                fprintf(fp,"%u ",edge_costs[offset]);
            }
            offset++;
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

unsigned int Graph::GetNumVertices()
{
    return num_verts;
}

unsigned int Graph::GetNumEdges()
{
    return num_edges;
}

unsigned int Graph::GetMaxDegree()
{
    return max_degree;
}

unsigned int *Graph::GetEdgeOffsets()
{
    return edge_offsets;
}

unsigned int *Graph::GetEdgeList()
{
    return edge_list;
}

unsigned int *Graph::GetEdgeCosts()
{
    return edge_costs;
}

unsigned int **Graph::GetEdgeOffsetsPtr()
{
    return &edge_offsets;
}

unsigned int **Graph::GetEdgeListPtr()
{
    return &edge_list;
}

unsigned int **Graph::GetEdgeCostsPtr()
{
    return &edge_costs;
}

int Graph::GetMetisGraphType()
{
    return graph_type;
}

unsigned int Graph::GetAdjacencyListLength()
{
    return adj_list_length;
}

// ****************************************************************************
//  Method:  Graph::GenerateSimpleKWayGraph
//
//  Purpose:
//      Generates a simple k-way tree from specified number of nodes and degree
//
//  Arguments:
//    verts: number of vertices in the graph
//    degree: specify k for k-way tree
//
//  Programmer:  Aditya Sarwade
//  Creation:    June 16, 2011
//
//  Modifications:
//
// ****************************************************************************
void Graph::GenerateSimpleKWayGraph(
    unsigned int verts,
    unsigned int degree)
{
	unsigned int index=0;
	unsigned int offset=0,j;
	unsigned int temp;

    if(edge_offsets==NULL)
    {
        if_delete_arrays=true;
        edge_offsets=new unsigned int[verts+1];
        edge_list=new unsigned int[verts*(degree+1)];
    }

	for(index=0;index<verts;index++)
	{
		edge_offsets[index]=offset;
		for(j=0;j<degree;j++)
		{
			temp=index*degree+(j+1);
			if(temp<verts)
			{
				edge_list[offset]=temp;
				offset++;
			}
		}
		if(index!=0)
		{
			edge_list[offset]=(unsigned int)floor(
					(float)(index-1)/(float)degree);
			offset++;
		}
	}

	//Add length of the adjacency list to last position
	edge_offsets[verts]=offset;

    adj_list_length=offset;
    num_edges=offset/2;
    num_verts=verts;
    graph_type=0;

    max_degree = degree + 1;
}

// ****************************************************************************
//  Method:  Graph::GetVertexLengths
//
//  Purpose:
//      Calculates the path lengths of each vertex from a specified source vetex
//
//  Arguments:
//    cost: array to return the path lengths for each vertex.
//    source: source vertex to calculate path lengths from.
//
//  Programmer:  Aditya Sarwade
//  Creation:    June 16, 2011
//
//  Modifications:
//
// ****************************************************************************
unsigned int * Graph::GetVertexLengths(
		unsigned int *cost,
		unsigned int source)
{
	//BFS uses Queue data structure
	for(int i=0;i<num_verts;i++)
		cost[i]=UINT_MAX;

	cost[source]=0;
	unsigned int nid;
	unsigned int next,offset;
	int n;
	unsigned int num_verts_visited=0;
	std::list<unsigned int> q;
	n=q.size();
	q.push_back(source);
	while(!q.empty())
	{
		n=q.front();
		num_verts_visited++;
		q.pop_front();
		offset=edge_offsets[n];
		next=edge_offsets[n+1];
		while(offset<next)
		{
			nid=edge_list[offset];
			offset++;
			if(cost[nid]>cost[n]+1)
			{
				cost[nid]=cost[n]+1;
				q.push_back(nid);
			}
		}
	}
	return cost;
}



}