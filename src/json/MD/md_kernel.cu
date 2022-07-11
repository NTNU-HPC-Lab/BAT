// Select which precision that are used in the calculations
// And define the replacements for the template inputs

// ****************************************************************************
// Function: compute_lj_force
//
// Purpose:
//   GPU kernel to calculate Lennard Jones force
//
// Arguments:
//      force3:     array to store the calculated forces
//      position:   positions of atoms
//      neighCount: number of neighbors for each atom to consider
//      neighList:  atom neighbor list
//      cutsq:      cutoff distance squared
//      lj1, lj2:   LJ force constants
//      inum:       total number of atoms
//
// Returns: nothing
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
extern "C" {

__global__ void compute_lj_force(float3* force3,
                                 float4* position,
                                 int neighCount,
                                 int* neighList,
                                 const float cutsq,
                                 const float lj1,
                                 const float lj2,
                                 const int inum)
{
    // Global ID - "WORK_PER_THREAD" atoms per thread
    int idx = (blockIdx.x*blockDim.x + threadIdx.x) * WORK_PER_THREAD;
    for (int i = 0; i < WORK_PER_THREAD; i++) {
        int threadId = idx + i;

        // Ensure that the current thread id is less than total number of elements
        if (threadId < inum) {
            // Position of this thread's atom
            float4 ipos = position[threadId];

            // Force accumulator

            float3 f = {0.0f, 0.0f, 0.0f};
            int j = 0;

            while (j < neighCount)
            {
		int jidx = 0;
                if (j*inum + threadId < inum) jidx = neighList[j*inum + threadId];
		else continue;
                float4 jpos = position[jidx];

                // Calculate distance
                float delx = ipos.x - jpos.x;
                float dely = ipos.y - jpos.y;
                float delz = ipos.z - jpos.z;
                float r2inv = delx*delx + dely*dely + delz*delz;

                // If distance is less than cutoff, calculate force
                // and add to accumulator
                if (r2inv < cutsq)
                {
                    r2inv = 1.0f/r2inv;
                    float r6inv = r2inv * r2inv * r2inv;
                    float force = r2inv*r6inv*(lj1*r6inv - lj2);
		    /*
                    f.x += delx * force;
                    f.y += dely * force;
                    f.z += delz * force;
		    */
                }
                j++;
            }
            force3[threadId] = f;

            // store the results
        }
    }
}

}
