//compile with hipcc -x hip jacobiHeatHIP.cpp
//or hipcc -x cuda jacobiHeatHIP.cpp for nvidia
//optimal flags -O3 -march=native -mtune=native but they don't do much
//tasted on an rx 6600xt, hip is suppose to support nvidia but I haven't tested it

#include <stdio.h>
#include <math.h>
#include "hip/hip_runtime.h"
#include <iostream>
#include <vector>

#define maxsize 12000
#define iterations 900
#define ROW 450
#define COL 352
#define start 100
#define accuracy 35

#define MAX_THREADS_PER_BLOCK 32

__global__ void calculateHeat(float* table1, float* table2, float* diff)
{
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= maxsize || col >= maxsize || row == 0 || row == maxsize-1 || col == 0 || col == maxsize-1)
        return;
    
    table2[row*maxsize+col] = 0.25*(table1[(row-1)*maxsize+col] + table1[(row+1)*maxsize+col] + table1[row*maxsize+col-1] + table1[row*maxsize+col+1]);
    float diff_temp = (table2[row*maxsize+col] - table1[row*maxsize+col]) * (table2[row*maxsize+col] - table1[row*maxsize+col]);
    atomicAdd(diff, diff_temp);

}

__global__ void initData(float* table, float* diff){
    table[ROW*maxsize+COL] = start;
    diff[0] = 0;
}

int main(int argc, char* argv[])
{
    int i, j, k;

    std::vector<float> table1;
    std::vector<float> table2;
    table1.resize(maxsize*maxsize);
    table2.resize(maxsize*maxsize);

    float diff;

    // pointer to GPU memory
    float *d_table1;
    float *d_table2;
    float *d_diff;

    // allocate GPU memory
    hipMalloc((void**)&d_table1, maxsize * maxsize * sizeof(float));
    hipMalloc((void**)&d_table2, maxsize * maxsize * sizeof(float));
    hipMalloc((void**)&d_diff, sizeof(float));

    hipMemset(d_table1, 0, maxsize * maxsize * sizeof(float));
    hipMemset(d_table2, 0, maxsize * maxsize * sizeof(float));
    hipMemset(d_diff, 0, sizeof(float));
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK);
    dim3 blocks((maxsize + threadsPerBlock.x - 1) / threadsPerBlock.x, (maxsize + threadsPerBlock.y - 1) / threadsPerBlock.y);
    std::cout << "blocks.x = " << blocks.x << std::endl;
    std::cout << "blocks.y = " << blocks.y << std::endl;
    std::cout << "threadsPerBlock.x = " << threadsPerBlock.x << std::endl;
    std::cout << "threadsPerBlock.y = " << threadsPerBlock.y << std::endl;

    /* repeate for each iteration */
    for(k = 0; k < iterations; k++){

        hipLaunchKernelGGL(initData, 1, 1, 0, 0, d_table1, d_diff);
        hipDeviceSynchronize();

        //call kernel
        hipLaunchKernelGGL(calculateHeat, blocks, threadsPerBlock, 0, 0, d_table1, d_table2, d_diff);

        //copy diff back to host
        hipMemcpy(&diff, d_diff, sizeof(float), hipMemcpyDeviceToHost);
        diff = sqrt(diff);
        // std::cout << "diff = " << diff << std::endl;
        /* print difference and check convergence */
        // printf("diff = %3.25f\n\n", diff);
        if (diff < accuracy) {
            printf("\n\nConvergence in %d iterations\n\n", k);
            printf("diff = %3.25f\n\n", diff);
            break;
        }

        //swap table1 and table2 in gpu
        std::swap(d_table1, d_table2);
    } 

    std::cout << "final diff = " << diff << std::endl;
    
    return 0;
}
