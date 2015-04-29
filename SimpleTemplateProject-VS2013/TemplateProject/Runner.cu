#include <cudaDefs.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <curand.h>
#include<curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

//FUNCTIONS
#define imin(a,b) (a<b?a:b)
#define imax(a,b) (a<b?b:a)

//#define SHARED 32

//POPULATION
#define NUM_VALS 2048
#define DIMENSION 16

//THREADS, BLOCKS
#define THREADS 64
#define FITNESS_THREADS 128
#define BLOCKS imin(8192, (NUM_VALS+THREADS-1)/THREADS)
#define FITNESS_BLOCKS imin(8192, (NUM_VALS+FITNESS_THREADS-1)/FITNESS_THREADS)
#define PRT_BLOCKS imin(8192, (DIMENSION+DIMENSION-1)/THREADS)

//SOLUTION
#define HI 500
#define LO 0

//CONFIG
#define CONF_PARMS 2
#define CONF_PATHLENGTH 0
#define CONF_STEP 1

#define ITERATION 20

//LEADER
__constant__ float leader[DIMENSION];
__constant__ float config[CONF_PARMS];

FILE * pFile;

void log(const char* type, float time) {
	fprintf(pFile, "(%s) elapsed: %1.3f\n", type, time);
}

#pragma region ERROR FUNCTION
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#pragma endregion

#pragma region REDUCTION
__device__ float atomicMaxf(float* address, float val)
{
	int *address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	while (val < __int_as_float(old)) {
		assumed = old;
		old = atomicCAS(address_as_int, assumed,
			__float_as_int(val));
	}
	return __int_as_float(old);
}
__global__ void max_reduce(float* values, int* max_index, float* maximum)
{
	__shared__ float shared[FITNESS_THREADS];

	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	shared[tid] = 0.0f;

	if (gid < NUM_VALS) {
		shared[tid] = values[gid];
		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
		{
			if (tid < s)
				shared[tid] = min(shared[tid], shared[tid + s]);  // 2
			__syncthreads();
		}

		__syncthreads();
		if (tid == 0) {
			maximum[blockIdx.x] = shared[tid];
			atomicMaxf(maximum, shared[0]);
		}
		__syncthreads();
		if (maximum[0] == values[gid]) {
			*max_index = gid;
		}
	}
	__syncthreads();
}
#pragma endregion

#pragma region RANDOM GENERATOR
__global__ void init_rand(curandState *state, size_t size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		curand_init(1337, idx, 0, &state[idx]);
}
__global__ void make_rand(curandState *state, float
	*randArray, const size_t size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		randArray[idx] = curand_uniform(&state[idx]);
}
#pragma endregion

#pragma region UTIL FUNCTION
float random_float()
{
	float f = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / (HI - LO));
	return f;
}
float max_value(float* values, const size_t size) {
	float vysledek = 0.0;
	for (int i = 0; i<size; i++) {
		if (values[i] > vysledek)
			vysledek = values[i];
	}
	return vysledek;
}
#pragma endregion

#pragma region POPULATION
void population_fill(float* arr)
{
	int index = 0;
	for (int i = 0; i < DIMENSION; ++i) {
		for (int j = 0; j < NUM_VALS; j++)
			arr[index++] = random_float();
	}
}
#pragma endregions
__global__ void eval(float* in, float* out) {
	unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
	float eval = 0.0;
	if (gid < NUM_VALS) {
		for (unsigned int i = 0; i < DIMENSION * NUM_VALS; i += NUM_VALS) {
			float val = in[gid + i];
			eval += -(in[gid + i]) * sin(sqrt(abs(in[gid + i])));
		}
		out[gid] = eval;
	}
	__syncthreads();
}
__global__ void step(float* in, float* temp, float* out, float* prt) {
	__shared__ float best_step[THREADS];
	__shared__ float minimum[THREADS];
	unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int lid = threadIdx.x;
	
	if (gid < NUM_VALS) {
		best_step[lid] = 0.0;
		minimum[lid] = out[gid];
		float s = config[CONF_STEP];
		float prt_val = 0.0;
		while (s < config[CONF_PATHLENGTH]) {
			//calculate next step parameters
			for (int i = 0; i < DIMENSION; i++) {
				//prt_val = ((prt[gid * DIMENSION + i] > 0.01) ? 1.0 : 0.0);
				//opravit, protože je naalokovano málo hodnot rand
				temp[gid + NUM_VALS * i] = in[gid + NUM_VALS * i] + (leader[i] - in[gid + NUM_VALS * i]) * s;
			}

			//evaluate solution
			float eval = 0.0;
			for (unsigned int i = 0; i < DIMENSION * NUM_VALS; i += NUM_VALS) {
				eval += -temp[gid + i] * sin(sqrt(abs(temp[gid + i])));
			}
			if (eval < minimum[lid]) {
				minimum[lid] = eval;
				best_step[lid] = s;
			}
			s += config[CONF_STEP];
		}
		
		//change solution to best
		for (int i = 0; i < DIMENSION; i++) {
			prt_val = ((prt[gid * DIMENSION + i]>0.2) ? 1.0 : 0.0);
			in[gid + NUM_VALS * i] = in[gid + NUM_VALS * i] + (leader[i] - in[gid + NUM_VALS * i]) * best_step[lid];
		}
		out[gid] = minimum[lid];
	}
	__syncthreads();
}

int main(int argc, char *argv[])
{
	pFile = fopen("log.txt", "a");
	srand(time(NULL));
	
	//int max_size = imin(THREADS*BLOCKS, real_size);
	float elapsedTime = 0.0, elapsedMemcpy = 0.0, elapsedKernel = 0.0;
	float *host_in, *host_out;
	float host_config[CONF_PARMS];
	host_config[CONF_STEP] = 0.0089;
	host_config[CONF_PATHLENGTH] = 2.31;
	cudaEvent_t startEvent, stopEvent;
	
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaMemcpyToSymbol(config, host_config, sizeof(float)* CONF_PARMS);

	printf("Velikost populace: %d\n", NUM_VALS);
	//printf("Max size = %d\n", max_size);

	cudaMallocHost(&host_in, NUM_VALS * DIMENSION * sizeof(float));
	cudaMallocHost(&host_out, NUM_VALS * sizeof(float));
	
	population_fill(host_in);

	float *device_in, *device_out, *device_maximum, *device_temp;
	float* host_leader = (float*)malloc(sizeof(float) * DIMENSION);
	int* device_leader_index;
	float* dev_prt_vector;
	curandState* dev_states;
	gpuErrchk(cudaMalloc((void**)&device_in, NUM_VALS * DIMENSION * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&device_temp, NUM_VALS * DIMENSION * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&device_out, NUM_VALS * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&device_leader_index, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&device_maximum, sizeof(float) * FITNESS_BLOCKS));
	gpuErrchk(cudaMalloc((void**)&dev_prt_vector, DIMENSION * NUM_VALS * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_states, DIMENSION * sizeof(curandState)));

	init_rand << <PRT_BLOCKS, THREADS >> >(dev_states, DIMENSION * NUM_VALS);

	//EVALUATE
	gpuErrchk(cudaMemcpy(device_in, host_in, sizeof(float)* DIMENSION * NUM_VALS, cudaMemcpyHostToDevice));
	eval << <BLOCKS, THREADS >> >(device_in, device_out);
	gpuErrchk(cudaMemcpy(host_out, device_out, sizeof(float)* NUM_VALS, cudaMemcpyDeviceToHost));
	
	//for (int i = 0; i < imin(NUM_VALS, 32); i++)
	//	printf("%1.3f\n", host_out[i]);
	
	//REDUCE
	max_reduce << <FITNESS_BLOCKS, FITNESS_THREADS >> >(device_out, device_leader_index, device_maximum);
	int* host_leader_index = (int*)malloc(sizeof(int));
	gpuErrchk(cudaMemcpy(host_leader_index, device_leader_index, sizeof(int), cudaMemcpyDeviceToHost));
	float* host_minimum = (float*)malloc(sizeof(float));
	gpuErrchk(cudaMemcpy(host_minimum, device_maximum, sizeof(float), cudaMemcpyDeviceToHost));
	printf("Minimum je %1.3f\n", host_minimum[0]);

	int index = 0;
	//printf("Leader: ");
	for (int i = host_leader_index[0]; i < DIMENSION * NUM_VALS; i += NUM_VALS) {
		host_leader[index++] = host_in[i];
		//printf("%1.3f,", host_leader[index-1]);
	}
	//printf("\n");
	cudaMemcpyToSymbol(leader, host_leader, sizeof(float) * DIMENSION);

	//float* host_prt_vector = (float*)malloc(sizeof(float) * DIMENSION);
	//gpuErrchk(cudaMemcpy(host_prt_vector, dev_prt_vector, sizeof(float)* DIMENSION, cudaMemcpyDeviceToHost));

	for (int iter = 0; iter < ITERATION; iter++) {
		make_rand << <PRT_BLOCKS, THREADS >> >(dev_states, dev_prt_vector, DIMENSION * NUM_VALS);
		//STEP
		gpuErrchk(cudaMemcpy(device_in, host_in, sizeof(float)* DIMENSION * NUM_VALS, cudaMemcpyHostToDevice));
		step << <BLOCKS, THREADS >> >(device_in, device_temp, device_out, dev_prt_vector);
		gpuErrchk(cudaMemcpy(host_out, device_out, sizeof(float)* NUM_VALS, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(host_in, device_in, sizeof(float)* DIMENSION * NUM_VALS, cudaMemcpyDeviceToHost));

		/*printf("---------------------------------------------------\n");
		for (int i = 0; i < imin(NUM_VALS, 10); i++)
			printf("%1.3f\n", host_out[i]);*/

		//REDUCE
		max_reduce << <FITNESS_BLOCKS, FITNESS_THREADS >> >(device_out, device_leader_index, device_maximum);
		gpuErrchk(cudaMemcpy(host_leader_index, device_leader_index, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(host_minimum, device_maximum, sizeof(float), cudaMemcpyDeviceToHost));
		printf("Minimum je %1.3f = %1.3f\n", host_out[*host_leader_index], host_minimum[0]);

		int index = 0;
		//printf("Leader: ");
		for (int i = host_leader_index[0]; i < DIMENSION * NUM_VALS; i += NUM_VALS) {
			host_leader[index++] = host_in[i];
			//printf("%1.3f,", host_leader[index - 1]);
		}
		//printf("\n");
		cudaMemcpyToSymbol(leader, host_leader, sizeof(float)* DIMENSION);
	}

	cudaFree(device_in);
	cudaFree(device_out);
	cudaFree(device_leader_index);
	cudaFree(device_maximum);
	cudaFree(dev_prt_vector);
	cudaFree(dev_states);
	cudaFree(device_temp);
	
	cudaFreeHost(host_in);
	cudaFreeHost(host_out);
	
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	//free(host_prt_vector);
	free(host_leader);
	free(host_leader_index);
	fclose(pFile);
}

