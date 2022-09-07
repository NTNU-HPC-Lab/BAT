#if VECTOR_TYPE == 1
    typedef float vector;
#elif VECTOR_TYPE == 2
    typedef float2 vector;
#elif VECTOR_TYPE == 4
    typedef float4 vector;
#endif // VECTOR_TYPE


inline __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}

inline __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

inline __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}

inline __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}

inline __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

inline __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

inline __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

inline __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}

inline __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline __device__ float2 rsqrtf(float2 x){
    return make_float2(rsqrtf(x.x), rsqrtf(x.y));
}

inline __device__ float4 rsqrtf(float4 x){
    return make_float4(rsqrtf(x.x), rsqrtf(x.y), rsqrtf(x.z), rsqrtf(x.w));
}

extern "C" {
// method to calculate acceleration caused by body J
__device__ void updateAcc(vector bodyAcc[3], float bodyPos[3], // position of body I
	vector bufferPosX, vector bufferPosY, vector bufferPosZ, vector bufferMass, // position and mass of body J
	float softeningSqr) // to avoid infinities and zero division
{
	#if USE_SOA == 0
	{
		float3 d;
		d.x = bufferPosX - bodyPos[0];
		d.y = bufferPosY - bodyPos[1];
		d.z = bufferPosZ - bodyPos[2];

		float distSqr = (d.x * d.x) + (d.y * d.y) + (d.z * d.z) + softeningSqr;
		float invDist = rsqrtf(distSqr);
		float f = bufferMass * invDist * invDist * invDist;

		bodyAcc[0] += d.x * f;
		bodyAcc[1] += d.y * f;
		bodyAcc[2] += d.z * f;
	}
	#else // USE_SOA != 0
	{
		vector distanceX = bufferPosX - bodyPos[0];
		vector distanceY = bufferPosY - bodyPos[1];
		vector distanceZ = bufferPosZ - bodyPos[2];

		vector invDist = rsqrtf(distanceX * distanceX + distanceY * distanceY + distanceZ * distanceZ + softeningSqr);
		vector f = bufferMass * invDist * invDist * invDist;

		bodyAcc[0] += distanceX * f;
		bodyAcc[1] += distanceY * f;
		bodyAcc[2] += distanceZ * f;
	}
	#endif // USE_SOA == 0
}

// method to calculate acceleration caused by body J
__device__ void updateAccGM(vector bodyAcc[3],
	float bodyPos[3], // position of body I
    float4* oldBodyInfo, // data; [X,Y,Z,mass]
    vector* oldPosX,
    vector* oldPosY,
    vector* oldPosZ,
    vector* mass,
	int index,
	float softeningSqr) // to avoid infinities and zero division
{
	#if USE_SOA == 0
	{
		updateAcc(bodyAcc, bodyPos,
			oldBodyInfo[index].x,
			oldBodyInfo[index].y,
			oldBodyInfo[index].z,
			oldBodyInfo[index].w,
			softeningSqr);
	}
	#else // USE_SOA != 0
	{
		updateAcc(bodyAcc, bodyPos,
			oldPosX[index],
			oldPosY[index],
			oldPosZ[index],
			mass[index],
			softeningSqr);
	}
	#endif // USE_SOA == 0
}


// method to load thread specific data from memory
__device__ void loadThreadData(
    float4* oldBodyInfo, // data; [X,Y,Z,mass]
    float* oldPosX,
    float* oldPosY,
    float* oldPosZ,
    float* mass,
    float4* oldVel, // velocity info
    float* oldVelX,
    float* oldVelY,
    float* oldVelZ,
    float bodyPos[][3],
    float bodyVel[][3],
    float* bodyMass) // thread data
{
	int index = (blockIdx.x*blockDim.x + threadIdx.x) * OUTER_UNROLL_FACTOR;
#if INNER_UNROLL_FACTOR2 > 0
	# pragma unroll INNER_UNROLL_FACTOR2
#endif
	for (int j = 0; j < OUTER_UNROLL_FACTOR; ++j) {
		#if USE_SOA == 0
		{
			// store 'thread specific' body info to registers
			bodyPos[j][0] = oldBodyInfo[index + j].x;
			bodyPos[j][1] = oldBodyInfo[index + j].y;
			bodyPos[j][2] = oldBodyInfo[index + j].z;

			bodyVel[j][0] = oldVel[index + j].x;
			bodyVel[j][1] = oldVel[index + j].y;
			bodyVel[j][2] = oldVel[index + j].z;

			bodyMass[j] = oldBodyInfo[index + j].w;
		}
		#else // USE_SOA != 0
		{
			// store 'thread specific' body info to registers
			bodyPos[j][0] = oldPosX[index + j];
			bodyPos[j][1] = oldPosY[index + j];
			bodyPos[j][2] = oldPosZ[index + j];

			bodyVel[j][0] = oldVelX[index + j];
			bodyVel[j][1] = oldVelY[index + j];
			bodyVel[j][2] = oldVelZ[index + j];

			bodyMass[j] = mass[index + j];
		}
		#endif // USE_SOA == 0
	}
}

// method will copy one item (X, Y, Z, mass) from input data to buffers
__device__ void fillBuffers(
    float4* oldBodyInfo, // global (input) data; [X,Y,Z,mass]
    vector* oldPosX,
    vector* oldPosY,
    vector* oldPosZ,
    vector* mass,
    vector bufferPosX[BLOCK_SIZE], // buffers
    vector bufferPosY[BLOCK_SIZE],
    vector bufferPosZ[BLOCK_SIZE],
    vector bufferMass[BLOCK_SIZE],
	int offset)
{
    int tid = threadIdx.x;
	#if USE_SOA == 0
	{
		bufferPosX[tid] = oldBodyInfo[offset + tid].x;
		bufferPosY[tid] = oldBodyInfo[offset + tid].y;
		bufferPosZ[tid] = oldBodyInfo[offset + tid].z;
		bufferMass[tid] = oldBodyInfo[offset + tid].w;
	}
	#else // USE_SOA != 0
	{
		bufferPosX[tid] = oldPosX[offset + tid];
		bufferPosY[tid] = oldPosY[offset + tid];
		bufferPosZ[tid] = oldPosZ[offset + tid];
		bufferMass[tid] = mass[offset + tid];
	}
	#endif // USE_SOA == 0
}


// kernel calculating new position and velocity for n-bodies
#if VECTOR_SIZE > 1
__global__ __attribute__((vec_type_hint(vector)))
#endif
__global__ void nbody_kernel(float timeDelta,
    float4* oldBodyInfo, // pos XYZ, mass
    vector* oldPosX,
    vector* oldPosY,
    vector* oldPosZ,
    vector* mass,
	float4* newBodyInfo,
    float4* oldVel, // XYZ, W unused
    vector* oldVelX,
    vector* oldVelY,
    vector* oldVelZ,
    float4* newVel, // XYZ, W set to 0.f
    float damping,
    float softeningSqr,
    int n)
{
    // buffers for bodies info processed by the work group
    __shared__ vector bufferPosX[BLOCK_SIZE];
    __shared__ vector bufferPosY[BLOCK_SIZE];
    __shared__ vector bufferPosZ[BLOCK_SIZE];
    __shared__ vector bufferMass[BLOCK_SIZE];

    // each thread holds a position/mass of the body it represents
    float bodyPos[OUTER_UNROLL_FACTOR][3];
    float bodyVel[OUTER_UNROLL_FACTOR][3];
    vector bodyAcc[OUTER_UNROLL_FACTOR][3];
    float bodyMass[OUTER_UNROLL_FACTOR];

	// clear acceleration
#if INNER_UNROLL_FACTOR2 > 0
	# pragma unroll INNER_UNROLL_FACTOR2
#endif
	for (int j = 0; j < OUTER_UNROLL_FACTOR; ++j) {
#if VECTOR_TYPE == 1
		bodyAcc[j][0] = bodyAcc[j][1] = bodyAcc[j][2] = .0f;
#elif VECTOR_TYPE == 2
        bodyAcc[j][0] = bodyAcc[j][1] = bodyAcc[j][2] = make_float2(.0f);
#elif VECTOR_TYPE == 4
        bodyAcc[j][0] = bodyAcc[j][1] = bodyAcc[j][2] = make_float4(.0f);
#endif
	}

	// load data
	loadThreadData(oldBodyInfo, ( float*)oldPosX, ( float*)oldPosY, ( float*)oldPosZ, ( float*)mass,
		oldVel, ( float*)oldVelX, ( float*)oldVelY, ( float*)oldVelZ, // velocity
		bodyPos, bodyVel, bodyMass); // values to be filled

	int blocks = n / (BLOCK_SIZE * VECTOR_TYPE); // each calculates effect of BLOCK_SIZE atoms to currect, i.e. thread's, one
	// start the calculation, process whole blocks

	for (int i = 0; i < blocks; i++) {
		#if LOCAL_MEM == 1
			// load new values to buffer.
			// We know that all threads can be used now, so no condition is necessary
			fillBuffers(oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass, bufferPosX, bufferPosY, bufferPosZ, bufferMass, i * BLOCK_SIZE);
			__syncthreads();
		#endif // LOCAL_MEM == 1

			// calculate the acceleration between the thread body and each other body loaded to buffer
        #if INNER_UNROLL_FACTOR1 > 0
		# pragma unroll INNER_UNROLL_FACTOR1
        #endif

		for(int index =  0; index < BLOCK_SIZE; index++) {
            #if INNER_UNROLL_FACTOR2 > 0
			# pragma unroll INNER_UNROLL_FACTOR2
            #endif
			for (int j = 0; j < OUTER_UNROLL_FACTOR; ++j) {
				#if LOCAL_MEM == 1
					updateAcc(bodyAcc[j], bodyPos[j],
						bufferPosX[index], bufferPosY[index], bufferPosZ[index], bufferMass[index],
						softeningSqr);

				#else // LOCAL_MEM != 1
					updateAccGM(bodyAcc[j], bodyPos[j],
						oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass,
						i * BLOCK_SIZE + index,
						softeningSqr);
				#endif // LOCAL_MEM == 1
			}
		}
        #if  LOCAL_MEM == 1
		__syncthreads(); // sync threads
        #endif
	}

	// sum elements of acceleration vector, if any
	float resAccX, resAccY, resAccZ;
	int index = (blockIdx.x*blockDim.x + threadIdx.x) * OUTER_UNROLL_FACTOR;
    #if INNER_UNROLL_FACTOR2 > 0
	# pragma unroll INNER_UNROLL_FACTOR2
    #endif
	for (int j = 0; j < OUTER_UNROLL_FACTOR; ++j) {
		resAccX = resAccY = resAccZ = 0.f;
		for (int i = 0; i < VECTOR_TYPE; i++)
		{
			resAccX += ((float*)&bodyAcc[j][0])[i];
			resAccY += ((float*)&bodyAcc[j][1])[i];
			resAccZ += ((float*)&bodyAcc[j][2])[i];
		}

		// 'export' result
		// calculate resulting position
		float resPosX = bodyPos[j][0] + timeDelta * bodyVel[j][0] + damping * timeDelta * timeDelta * resAccX;
		float resPosY = bodyPos[j][1] + timeDelta * bodyVel[j][1] + damping * timeDelta * timeDelta * resAccY;
		float resPosZ = bodyPos[j][2] + timeDelta * bodyVel[j][2] + damping * timeDelta * timeDelta * resAccZ;
		newBodyInfo[index + j] = make_float4(resPosX, resPosY, resPosZ, bodyMass[j]);
		// calculate resulting velocity
		float resVelX = bodyVel[j][0] + timeDelta * resAccX;
		float resVelY = bodyVel[j][1] + timeDelta * resAccY;
		float resVelZ = bodyVel[j][2] + timeDelta * resAccZ;
		newVel[index + j] = make_float4(resVelX, resVelY, resVelZ, 0.f);
	}
}



} // Extern C

/*
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int main(void)
{
    int N = 131072;
    float* oldBodyInfo = (float*)malloc(4*N*sizeof(float));
    float* newBodyInfo = (float*)malloc(4*N*sizeof(float));
    float* newVel = (float*)malloc(4*N*sizeof(float));
    float* oldVel = (float*)malloc(4*N*sizeof(float));

    float* oldPosX = (float*)malloc(N*sizeof(float));
    float* oldPosY = (float*)malloc(N*sizeof(float));
    float* oldPosZ = (float*)malloc(N*sizeof(float));
    float* mass = (float*)malloc(N*sizeof(float));
    float* oldVelX = (float*)malloc(N*sizeof(float));
    float* oldVelY = (float*)malloc(N*sizeof(float));
    float* oldVelZ = (float*)malloc(N*sizeof(float));

    oldBodyInfo[0] = 1.0f;


    for (int i = 0; i < N; i++) {
        oldPosX[i] = 1.0f;
        oldPosY[i] = 2.0f;
        oldPosZ[i] = 2.0f;
        mass[i] = 2.0f;

        oldVelX[i] = 1.0f;
        oldVelY[i] = 2.0f;
        oldVelZ[i] = 2.0f;

        oldBodyInfo[4*i] = oldPosX[i];
        oldBodyInfo[4*i + 1] = oldPosY[i];
        oldBodyInfo[4*i + 2] = oldPosZ[i];
        oldBodyInfo[4*i + 3] = mass[i];

        oldVel[4*i] = oldVelX[i];
        oldVel[4*i + 1] = oldVelY[i];
        oldVel[4*i + 2] = oldVelZ[i];
        oldVel[4*i + 3] = 0.0f;
    }

    float *d_oldBodyInfo, *d_newBodyInfo, *d_newVel, *d_oldVel;
    float *d_oldPosX, *d_oldPosY, *d_oldPosZ, *d_mass, *d_oldVelX, *d_oldVelY, *d_oldVelZ;

    cudaMalloc(&d_newBodyInfo, 4*N*sizeof(float));
    cudaMalloc(&d_newVel, 4*N*sizeof(float));
    cudaMalloc(&d_oldBodyInfo, 4*N*sizeof(float));
    cudaMalloc(&d_oldVel, 4*N*sizeof(float));

    cudaMalloc(&d_oldPosX, N*sizeof(float));
    cudaMalloc(&d_oldPosY, N*sizeof(float));
    cudaMalloc(&d_oldPosZ, N*sizeof(float));
    cudaMalloc(&d_mass, N*sizeof(float));
    cudaMalloc(&d_oldVelX, N*sizeof(float));
    cudaMalloc(&d_oldVelY, N*sizeof(float));
    cudaMalloc(&d_oldVelZ, N*sizeof(float));

    cudaMemcpy(d_oldBodyInfo, oldBodyInfo, 4*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_newBodyInfo, newBodyInfo, 4*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldVel, oldVel, 4*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_newVel, newVel, 4*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_oldPosX, oldPosX, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldPosY, oldPosY, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldPosZ, oldPosZ, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldVelX, oldVelX, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldVelY, oldVelY, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldVelZ, oldVelZ, N*sizeof(float), cudaMemcpyHostToDevice);

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int n = N;
    float damping = 0.5f;
    float softeningSqr = 0.01f;
    float timeDelta = 0.001f;

  nbody_kernel<<<grid, BLOCK_SIZE>>>(timeDelta, (float4*) oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass,
                                     (float4*)newBodyInfo, (float4*)oldVel, oldVelX, oldVelY, oldVelZ, (float4*)newVel,
                                     damping, softeningSqr, n);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
    return 0;

}

*/


