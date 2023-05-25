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
    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        std::cout << "Reading error\n";
        return 4;
    }

    // setup GPU memory
    char* fileDataGpu;
    int* matchGpu;
    int* total_matchesGpu;
    char* patternGpu;

    hipMalloc((void**)&fileDataGpu, fileSize * sizeof(char));
    hipMalloc((void**)&matchGpu, fileSize * sizeof(int));
    hipMalloc((void**)&total_matchesGpu, sizeof(int));
    hipMalloc((void**)&patternGpu, patternSize * sizeof(char));

    hipMemset(matchGpu, 0, fileSize * sizeof(int));
    hipMemset(total_matchesGpu, 0, sizeof(int));

    // send data to GPU
    hipMemcpy(fileDataGpu, buffer.data(), fileSize * sizeof(char), hipMemcpyHostToDevice);
    hipMemcpy(patternGpu, pattern.c_str(), patternSize * sizeof(char), hipMemcpyHostToDevice);

    // setup execution parameters
    dim3 threadsPerBlock(512, 1, 1);
    dim3 blocks((fileSize + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    // launch kernel
    hipLaunchKernelGGL(StringMatching, blocks, threadsPerBlock, 0, 0, fileDataGpu, fileSize, patternGpu, patternSize, matchGpu, fileSize, total_matchesGpu);

    // get results back
    std::vector<int> match(fileSize);
    int total_matches;
    hipMemcpy(match.data(), matchGpu, fileSize * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(&total_matches, total_matchesGpu, sizeof(int), hipMemcpyDeviceToHost);

	//print results
    std::cout << "Total matches: " << total_matches << std::endl;
	// for (int i=0; i<fileSize; i++) {
	// 	if (match[i] == 1) {
	// 		std::cout << "Match at index " << i << std::endl;
	// 	}
	// }

    // cleanup
    hipFree(fileDataGpu);
    hipFree(matchGpu);
    hipFree(total_matchesGpu);
    hipFree(patternGpu);
    file.close();

    return 0;
}
