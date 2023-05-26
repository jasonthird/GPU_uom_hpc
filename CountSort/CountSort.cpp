//compile with hipcc -x hip CountSort.cpp
//or hipcc -x cuda CountSort.cpp for nvidia
//optimal flags -O3 -march=native -mtune=native but they don't do much
//tasted on an rx 6600xt, hip is suppose to support nvidia but I haven't tested it

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <time.h>

__global__ void countSort(int* a,int n, int* tempArray){

    int i = blockIdx.x * blockDim.x + threadIdx.x;  

    if (i >= n)
        return;

    int count = 0;
    // for (int j = 0; j < n; j++)
    //     if (a[j] < a[i])
    //         count++;
    //     else if (a[j] == a[i] && j < i)
    //         count++;
    #pragma unroll //unroll the loop to increase performance, makes a small difference
    for (int j = 0; j < n; j++){
        count+= (a[j] < a[i]);
        count+= (a[j] == a[i] && j < i);
    }

    tempArray[count] = a[i];
}

int main(int args, char** argv){
    int n;
    if (args == 2){
        n = atoi(argv[1]);
    }
    else{
        std::cout << "Usage: ./CountSort <number of elements>" << std::endl;
        return 0;
    }
    std::vector<int> a;
    a.resize(n);
    srand((unsigned) time(NULL));
    for (int i = 0; i<n; i++){
        a[i] = (rand() % (1000 + 1));
    }

    // std::cout << "Unsorted array: ";
    // for (int i = 0; i < n; i++)
    //     std::cout << a[i] << " ";
    // std::cout << std::endl;

    //selecting GPU with highest Gflops/s
    //this is not necessary, but its good to know which GPU is fastest
    int dev = 0;
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);
    std::cout << "Device: " << deviceProp.name << std::endl << "Device number: " << dev << std::endl;
    hipSetDevice(0);

    //setup memory
    int* d_a;
    int* tempArray;
    hipMalloc((void**)&tempArray, n * sizeof(int));
    hipMalloc((void**)&d_a, n * sizeof(int));
    //hipMemset(tempArray, 0, n * sizeof(int));
    hipMemcpy(d_a, a.data(), n * sizeof(int), hipMemcpyHostToDevice);

    //setup execution parameters
    dim3 threadsPerBlock(512, 1, 1);
    dim3 blocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    //launch kernel
    hipLaunchKernelGGL(countSort, blocks, threadsPerBlock, 0, 0, d_a, n, tempArray);
    hipMemcpy(a.data(), tempArray, n * sizeof(int), hipMemcpyDeviceToHost);

    //cleanup
    hipFree(d_a);
    hipFree(tempArray);


    // std::cout << "Sorted array: ";
    // for (int i = 0; i < n; i++)
    //     std::cout << a[i] << " ";
    // std::cout << std::endl;

    //check if the array is sorted
    if (std::is_sorted(a.begin(), a.end()))
        std::cout << "Array is sorted" << std::endl;
    else
        std::cout << "Array is not sorted" << std::endl;

    return 0;
}