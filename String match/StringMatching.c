#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>

__global__ void StringMatching(char* fileData,long fileSize, char* pattern, int patternSize, int* match, int matchSize,int *total_matches){

    int i = blockIdx.x * blockDim.x + threadIdx.x;  

    if (i >= matchSize)
		return;
	
	int j;
	for (j = 0; j < patternSize; j++){
		if (fileData[i+j] != pattern[j])
			break;
	}
	if (j == patternSize){
		match[i] = 1;
		atomicAdd(total_matches, 1);
	}
}

int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size;
	char * buffer;
	char * filename;
	size_t result;
	int *match;
	int *total_matches;
	char *pattern;
	int patternSize;

	if (argc != 3) {
		printf ("Usage : %s <file_name> <string>\n", argv[0]);
		return 1;
	}
	filename = argv[1];
	pattern = argv[2];
	patternSize = strlen(pattern);
	pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	file_size = ftell (pFile);
	rewind (pFile);
	printf("file size is %ld\n", file_size);
	
	// allocate memory to contain the file:
	buffer = (char*) malloc (sizeof(char)*file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	// copy the file into the buffer:
	result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;}
    
    //setup gpu memory
    char * fileDataGpu;
    int * matchGpu;
	int * total_matchesGpu;
	char * patternGpu;

    hipMalloc((void**)&fileDataGpu, file_size * sizeof(char));
	hipMalloc((void**)&matchGpu, file_size * sizeof(int));
	hipMalloc((void**)&total_matchesGpu, sizeof(int));
	hipMalloc((void**)&patternGpu, patternSize * sizeof(char));

	hipMemset(matchGpu,0, file_size*sizeof(int));
	hipMemset(total_matchesGpu,0, sizeof(int));

    // //send data to gpu
    hipMemcpy(fileDataGpu, buffer, file_size * sizeof(char), hipMemcpyHostToDevice);
	// hipMemcpy(patternGpu, pattern, patternSize * sizeof(char), hipMemcpyHostToDevice);
	

    // //setup execution parameters
    // dim3 threadsPerBlock(512, 1, 1);
    // dim3 blocks((file_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
    
    // //launch kernel
    // hipLaunchKernelGGL(StringMatching, blocks, threadsPerBlock, 0, 0, fileDataGpu, file_size, patternGpu, patternSize, matchGpu, file_size, total_matchesGpu);

    
    // //get results back
    // hipMemcpy(match, matchGpu, file_size * sizeof(int), hipMemcpyDeviceToHost);
	// hipMemcpy(total_matches, total_matchesGpu, sizeof(int), hipMemcpyDeviceToHost);

	// printf("Total matches: %d\n", *total_matches);
	
	// //cleanup
	// hipFree(fileDataGpu);
	// hipFree(matchGpu);
	// hipFree(total_matchesGpu);
	// hipFree(patternGpu);
	// fclose (pFile);
	// free (buffer);

	return 0;
}