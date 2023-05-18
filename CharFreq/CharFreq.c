//compile with hipcc -x hip CountSort.cpp
//or hipcc -x cuda CountSort.cpp for nvidia
//optimal flags -O3 -march=native -mtune=native but they don't do much
//tasted on an rx 6600xt, hip is suppose to support nvidia but I haven't tested it

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>

#define N 128
#define base 0

__global__ void charFreq(char* fileData,long fileSize, int* freq){

    int i = blockIdx.x * blockDim.x + threadIdx.x;  

    if (i >= fileSize)
        return;

    atomicAdd(&freq[fileData[i] - base], 1);
}

int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size;
	char * buffer;
	char * filename;
	size_t result;
	int *freq;

    if (argc != 2) {
    printf ("Usage : %s <file_name>\n", argv[0]);
    return 1;
    }
	filename = argv[1];
	pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	freq = (int*) malloc (sizeof(int)*N);

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
    int * freqDataGpu;

    hipMalloc((void**)&fileDataGpu, file_size * sizeof(char));
    hipMalloc((void**)&freqDataGpu, N*sizeof(int));

    hipMemset(freqDataGpu,0, N*sizeof(int));

    //send file to gpu
    hipMemcpy(fileDataGpu, buffer, file_size * sizeof(char), hipMemcpyHostToDevice);

    //setup execution parameters
    dim3 threadsPerBlock(512, 1, 1);
    dim3 blocks((file_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
    
    //launch kernel
    hipLaunchKernelGGL(charFreq, blocks, threadsPerBlock, 0, 0, fileDataGpu, file_size, freqDataGpu);
    
    //get results back
    hipMemcpy(freq,freqDataGpu,N*sizeof(int),hipMemcpyDeviceToHost);


	for (int j=0; j<N; j++){
		printf("%d = %d\n", j+base, freq[j]);
	}	

	fclose (pFile);
	free (buffer);

	return 0;
}