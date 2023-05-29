#include <stdio.h>
#include <math.h>
#include "hip/hip_runtime.h"
#include <iostream>
#include <vector>

#define maxsize 500
#define iterations 900
#define ROW 50
#define COL 50
#define start 100
#define accuracy 27

#define MAX_THREADS_PER_BLOCK 32

__global__ void calculateHeat(double* table1, double* table2, double* diff)
{
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= maxsize || col >= maxsize || row == 0 || row == maxsize-1 || col == 0 || col == maxsize-1)
        return;

    table2[row*maxsize+col] = 0.25*(table1[(row-1)*maxsize+col] + table1[(row+1)*maxsize+col] + table1[row*maxsize+col-1] + table1[row*maxsize+col+1]);

    double diff_temp = (table2[row*maxsize+col] - table1[row*maxsize+col]) * (table2[row*maxsize+col] - table1[row*maxsize+col]);
    atomicAdd(diff, diff_temp);
    

}

__global__ void initData(double* table, double* diff){
    table[ROW*maxsize+COL] = start;
    diff[0] = 0;
}

int main(int argc, char* argv[])
{
    int i, j, k;

    std::vector<double> table1;
    std::vector<double> table2;
    table1.resize(maxsize*maxsize);
    table2.resize(maxsize*maxsize);

    double diff;

    // pointer to GPU memory
    double *d_table1;
    double *d_table2;
    double *d_diff;

    // allocate GPU memory
    hipMalloc((void**)&d_table1, maxsize * maxsize * sizeof(double));
    hipMalloc((void**)&d_table2, maxsize * maxsize * sizeof(double));
    hipMalloc((void**)&d_diff, sizeof(double));

    hipMemset(d_table1, 0, maxsize * maxsize * sizeof(double));
    hipMemset(d_table2, 0, maxsize * maxsize * sizeof(double));
    hipMemset(d_diff, 0, sizeof(double));
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK, 1);
    dim3 blocks((maxsize + threadsPerBlock.x - 1) / threadsPerBlock.x, (maxsize + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    /* repeate for each iteration */
    for(k = 0; k < iterations; k++){

        hipLaunchKernelGGL(initData, 1, 1, 0, 0, d_table1, d_diff);

        //call kernel
        hipLaunchKernelGGL(calculateHeat, threadsPerBlock, blocks, 0, 0, d_table1, d_table2, d_diff);

        //copy diff back to host
        hipMemcpy(&diff, d_diff, sizeof(double), hipMemcpyDeviceToHost);
        diff = sqrt(diff);
        std::cout << "diff = " << diff << std::endl;
        /* print difference and check convergence */
        // printf("diff = %3.25f\n\n", diff);
        if (diff < accuracy) {
            printf("\n\nConvergence in %d iterations\n\n", k);
            break;
        }

        //swap table1 and table2 in gpu
        std::swap(d_table1, d_table2);
    } 

    std::cout << "final diff = " << diff << std::endl;
    
    return 0;
}
