//compile with hipcc -x hip StringMatching.cpp
//or hipcc -x cuda StringMatching.cpp for NVIDIA
//optimal flags -O3 -march=native -mtune=native but they don't do much
//tasted on an RX 6600 XT, HIP is supposed to support NVIDIA but I haven't tested it

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <hip/hip_runtime.h>

#define MaxPatternSize 256

__global__ void StringMatching(char* fileData, long fileSize, char* pattern, int patternSize, int* match, int matchSize, int* total_matches) {
    __shared__ char sharedPattern[MaxPatternSize];

    // Load pattern into shared memory
	int tid = threadIdx.x;
    if (tid < patternSize) {
        sharedPattern[tid] = pattern[tid];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= matchSize)
        return;

    int j;
    int localMatch = 1; // Local variable to track matching status
    for (j = 0; j < patternSize; j++) {
        if (fileData[i + j] != sharedPattern[j]) {
            localMatch = 0;
            break;
        }
    }
        // Use atomicAdd with total_matches
    if (localMatch) {
		match[i]=1;
        atomicAdd(total_matches, 1);
    }
}



int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <file_name> <string>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string pattern = argv[2];
    int patternSize = pattern.size();

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "File error\n";
        return 2;
    }

    // obtain file size
    file.seekg(0, std::ios::end);
    long fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::cout << "File size is " << fileSize << std::endl;

    // allocate memory to contain the file
    char* buffer;
    hipHostMalloc((void**)&buffer, fileSize * sizeof(char));

    //read file
    file.read(buffer, fileSize);

    // setup GPU memory
    char* fileDataGpu;
    int* matchGpu;
    int* total_matchesGpu;
    char* patternGpu;

    hipMalloc((void**)&fileDataGpu, fileSize * sizeof(char));
    hipMalloc((void**)&patternGpu, patternSize * sizeof(char));
    hipMalloc((void**)&matchGpu, fileSize * sizeof(int));
    hipMalloc((void**)&total_matchesGpu, sizeof(int));

    // create a stream for async operations
    hipStream_t stream;
    hipStreamCreate(&stream);

    // send data to GPU
    hipMemcpyAsync(fileDataGpu, buffer, fileSize * sizeof(char), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(patternGpu, pattern.c_str(), patternSize * sizeof(char), hipMemcpyHostToDevice, stream);

    // zero out match array
    hipMemset(matchGpu, 0, fileSize * sizeof(int));
    hipMemset(total_matchesGpu, 0, sizeof(int));

    // setup execution parameters
    dim3 threadsPerBlock(512, 1, 1);
    dim3 blocks((fileSize + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    // launch kernel
    hipLaunchKernelGGL(StringMatching, blocks, threadsPerBlock, 0, stream, fileDataGpu, fileSize, patternGpu, patternSize, matchGpu, fileSize, total_matchesGpu);

    // get results back
    std::vector<int> match(fileSize);
    int total_matches;
    //  using async version of memcpy
    hipMemcpyAsync(match.data(), matchGpu, fileSize * sizeof(int), hipMemcpyDeviceToHost, stream);
    hipMemcpyAsync(&total_matches, total_matchesGpu, sizeof(int), hipMemcpyDeviceToHost, stream);


    // wait for all asynchronous operations to complete
    hipStreamSynchronize(stream);

	// print results
    std::cout << "Total matches: " << total_matches << std::endl;
	// for (int i=0; i<fileSize; i++) {
	// 	if (match[i] == 1) {
	// 		std::cout << "Match at index " << i << std::endl;
	// 	}
	// }

    // cleanup
    hipStreamDestroy(stream);
    hipFree(fileDataGpu);
    hipFree(matchGpu);
    hipFree(total_matchesGpu);
    hipFree(patternGpu);
    file.close();

    return 0;
}
