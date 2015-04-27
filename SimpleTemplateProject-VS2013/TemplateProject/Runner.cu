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
#define NUM_VALS_BLOCK 64
#define NUM_VALS 256
#define DIMENSION 1024

//THREADS, BLOCKS
#define THREADS 64
#define FITNESS_THREADS 128
#define BLOCKS imin(8192, (NUM_VALS_BLOCK+THREADS-1)/THREADS)
#define FITNESS_BLOCKS imin(8192, (NUM_VALS+FITNESS_THREADS-1)/FITNESS_THREADS)
#define PRT_BLOCKS imin(8192, (DIMENSION+DIMENSION-1)/THREADS)

//SOLUTION
#define RANDOM_MAXIMUM 500

//CONFIG
#define CONF_PARMS 2
#define CONF_PATHLENGTH 0
#define CONF_STEP 1

#define ITERATION 1000

//LEADER
//__constant__ best_indiv leader[1];
__constant__ float config[DIMENSION];

FILE * pFile;

void log(const char* type, float time) {
	fprintf(pFile, "(%s) elapsed: %1.3f\n", type, time);
}

//DEBUG ERRORS
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
	while (val > __int_as_float(old)) {
		assumed = old;
		old = atomicCAS(address_as_int, assumed,
			__float_as_int(val));
	}
	return __int_as_float(old);
}
__global__ void max_reduce(const float* values, int* max_index, float* maximum, 
	const size_t elements)
{
	__shared__ float shared[FITNESS_THREADS];

	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	shared[tid] = 0.0f;

	if (gid < elements)
		shared[tid] = values[gid];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s && gid < elements)
			shared[tid] = max(shared[tid], shared[tid + s]);  // 2
		__syncthreads();
	}
	
	__syncthreads();
	if (tid == 0) {
		maximum[blockIdx.x] = shared[tid];
		atomicMaxf(maximum, shared[0]);
	}	
	__syncthreads();
	if (gid < elements && maximum[0] == values[gid]) {
		__syncthreads();
		if (gid < elements) {
			*max_index = gid;
		}
	}
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
	return (float)(rand() % RANDOM_MAXIMUM) + 1.0;
}
float random_int()
{
	return (int)(rand() % RANDOM_MAXIMUM) + 1.0;
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
void population_fill(float* arr, const size_t size)
{
	int index = 0;
	int all = size;

	int result = imin(NUM_VALS_BLOCK, all);
	all -= NUM_VALS_BLOCK;

	while (result > 0) {
		for (int i = 0; i < DIMENSION; ++i) {
			for (int j = 0; j < result; j++)
				arr[index++] = random_float();
		}
		result = imin(NUM_VALS_BLOCK, all);
		all -= result;
	}
}
#pragma endregions

#pragma region SOMA EVOLUTION
/*
__global__ void step(const individual* pop, individual* new_pop, float* prt, size_t size) {
	int lid = threadIdx.x;
	int gid = blockIdx.x * THREADS + threadIdx.x;
	
	//for (float j = config[CONF_STEP]; j < config[CONF_PATHLENGTH]; j += config[CONF_STEP]) {
		#pragma unroll
		for (int i = 0; i < DIMENSION; i++) {
			if (prt[i] > 0.5) {
				new_pop[gid].val[i] = (leader[0].val[i] - pop[gid].val[i]) * config[CONF_STEP] + pop[gid].val[i];
			} else {
				new_pop[gid].val[i] = pop[gid].val[i];
			}
		}
	//}
	__syncthreads();
}
__global__ void evaluate(individual* individuals, const size_t size)
{
	int tid = blockIdx.x * THREADS + threadIdx.x;
	while (tid < size)
	{
		individuals[tid].fitness = (individuals[tid].val[0] * individuals[tid].val[1] * individuals[tid].val[2]);
		tid += THREADS * BLOCKS;
	}
	__syncthreads();
}
*/
/*
void soma(float *values, const size_t size)
{
	dim3 blocks(BLOCKS, 1);
	dim3 threads(THREADS, 1);
	
	//DEVICE
	float* dev_fitness;
	best_indiv* dev_leader;
	float* dev_prt_vector;
	curandState* dev_states;
	individual* dev_indiv;
	individual* dev_new_pop;

	float host_config[CONF_PARMS];
	host_config[CONF_PATHLENGTH] = 10.0;
	host_config[CONF_STEP] = 1.1;

	//int size = s * (int)(host_config[CONF_PATHLENGTH] / host_config[CONF_STEP]);

	gpuErrchk(cudaMalloc((void**)&dev_fitness, size * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_leader, sizeof(best_indiv)));
	gpuErrchk(cudaMalloc((void**)&dev_new_pop, size * sizeof(individual)));

	best_indiv* host_leader = (best_indiv*)malloc(sizeof(best_indiv));
	individual* population = (individual*)malloc(size * sizeof(individual));

	#pragma region SOMA PARAMETER INITIALIZE
	cudaMemcpyToSymbol(config, host_config, sizeof(float)* CONF_PARMS);
	#pragma endregion

	#pragma region INITIALIZE
	gpuErrchk(cudaMalloc((void**)&dev_prt_vector, DIMENSION * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_states, DIMENSION * sizeof(curandState)));

	init_rand << <1, threads >> >(dev_states, DIMENSION);
	make_rand << <1, threads >> >(dev_states, dev_prt_vector, DIMENSION);
	#pragma endregion
	
	#pragma region POPULATION TO GPU
	cudaMalloc((void**)&dev_indiv, size * sizeof(individual));
	cudaMemcpy(dev_indiv, values, size * sizeof(individual), cudaMemcpyHostToDevice);
	#pragma endregion

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	float sumEval = 0.0;
	float sumReduce = 0.0;
	float sumRandom = 0.0;
	float sumStep = 0.0;

	for (int iter = 0; iter < ITERATION; iter++) {

		cudaEventRecord(startEvent, 0);
		evaluate << <blocks, threads >> >(dev_indiv, size);
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		log("evaluat", elapsedTime);
		sumEval += elapsedTime;

		cudaEventRecord(startEvent, 0);
		max_reduce << <blocks, threads >> >(dev_indiv, dev_fitness, dev_leader, size);
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		log("reduce", elapsedTime);
		sumReduce += elapsedTime;
		gpuErrchk(cudaMemcpy(host_leader, dev_leader, sizeof(best_indiv), cudaMemcpyDeviceToHost));

		//printf("Leader: (%1.3f, %1.3f, %1.3f)(%d)\n", host_leader[0].val[0], host_leader[0].val[1], host_leader[0].val[2], host_leader[0].index);
		cudaMemcpyToSymbol(leader, host_leader, sizeof(best_indiv));

		cudaEventRecord(startEvent, 0);
		make_rand << <1, threads >> >(dev_states, dev_prt_vector, DIMENSION);
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		log("random", elapsedTime);
		sumRandom += elapsedTime;
		
		cudaEventRecord(startEvent, 0);
		step << <blocks, threads >> >(dev_indiv, dev_new_pop, dev_prt_vector, size);
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		log("step", elapsedTime);
		sumStep += elapsedTime;

		gpuErrchk(cudaMemcpy(population, dev_indiv, size * sizeof(individual), cudaMemcpyDeviceToHost));
		//for (int i = 0; i < 10; i++)
			//printf("%1.3f, %1.3f, %1.3f = %1.3f\n", population[i].val[0], population[i].val[1], population[i].val[2], population[i].fitness);
		//printf("\n");

		individual * temp = dev_new_pop;
		dev_new_pop = dev_indiv;
		dev_indiv = temp;
	}

	printf("Avg Evaluated: %1.3f\n", sumEval / ITERATION);
	printf("Avg Reduced: %1.3f\n", sumReduce / ITERATION);
	printf("Avg Random: %1.3f\n", sumRandom / ITERATION);
	printf("Avg Step: %1.3f\n", sumStep / ITERATION);
	
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	free(host_leader);
	free(population);

	cudaFree(dev_fitness);
	cudaFree(dev_leader);
	cudaFree(dev_prt_vector);
	cudaFree(dev_states);
	cudaFree(dev_indiv);
	cudaFree(dev_new_pop);
}
*/
#pragma endregion

__global__ void operace(float* in, float* out) {
	__shared__ float temp[NUM_VALS_BLOCK];

	unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int lid = threadIdx.x;
	temp[lid] = 0;	//copy to shared
	
	if (gid < NUM_VALS_BLOCK) {
		#pragma unroll
		for (unsigned int i = 0; i < DIMENSION * NUM_VALS_BLOCK; i += NUM_VALS_BLOCK)
		{
			temp[lid] += in[gid + i];
		}
		out[gid] = temp[lid];
	}
	__syncthreads();
}

bool test(float* values, unsigned int size) {
	float test = values[0];
	for (int i = 1; i < size; i++)
	if (values[i] != test)
		return false;
	return true;
}

int main(int argc, char *argv[])
{
	pFile = fopen("log.txt", "a");
	srand(time(NULL));
	
	int real_size = NUM_VALS;
	int max_size = imin(NUM_VALS_BLOCK, real_size);
	float elapsedTime = 0.0, elapsedMemcpy = 0.0, elapsedKernel = 0.0;
	float *host_in, *host_out;
	cudaEvent_t startEvent, stopEvent;
	
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	printf("Velikost populace: %d\n", real_size);
	printf("Max size = %d\n", max_size);

	cudaMallocHost(&host_in, real_size * DIMENSION * sizeof(float));
	cudaMallocHost(&host_out, real_size * sizeof(float));
	
	population_fill(host_in, real_size);
	real_size -= max_size;

	float *device_in, *device_out, *device_maximum;
	float* host_leader = (float*)malloc(sizeof(float) * DIMENSION);
	int* device_leader_index;
	float* host_in_start = host_in;
	float* host_out_start = host_out;
	float* dev_prt_vector;
	curandState* dev_states;

	gpuErrchk(cudaMalloc((void**)&device_in, max_size * DIMENSION * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&device_out, max_size * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&device_leader_index, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&device_maximum, FITNESS_BLOCKS * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_prt_vector, DIMENSION * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_states, DIMENSION * sizeof(curandState)));

	while (max_size > 0) {
		cudaEventRecord(startEvent, 0);
		gpuErrchk(cudaMemcpy(device_in, host_in_start, sizeof(float)* DIMENSION * max_size, cudaMemcpyHostToDevice));
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		elapsedMemcpy += elapsedTime;

		cudaEventRecord(startEvent, 0);
		operace<< <BLOCKS, THREADS >> >(device_in, device_out);
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		elapsedKernel += elapsedTime;
	
		gpuErrchk(cudaMemcpy(host_out_start, device_out, sizeof(float)* max_size, cudaMemcpyDeviceToHost));

		host_in_start += max_size*DIMENSION;
		host_out_start += max_size;
		max_size = imin(real_size, NUM_VALS_BLOCK);
		real_size -= max_size;
	}
	
	printf("Memcpy: %1.3f\n", elapsedMemcpy);
	printf("Kernel: %1.3f\n", elapsedKernel);
	
	for (int i = 0; i < imin(NUM_VALS, 16); i++)
		printf("%1.3f\n", host_out[i]);
	
	cudaEventRecord(startEvent, 0);
	max_reduce << <FITNESS_BLOCKS, FITNESS_THREADS >> >(device_out, device_leader_index, device_maximum, NUM_VALS);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Redukce: %1.3f\n", elapsedTime);
	
	int* host_leader_index = (int*)malloc(sizeof(int));
	gpuErrchk(cudaMemcpy(host_leader_index, device_leader_index, sizeof(int), cudaMemcpyDeviceToHost));

	int index = 0;
	for (int i = host_leader_index[0]; i < DIMENSION * NUM_VALS; i += DIMENSION) {
		host_leader[index++] = host_in[i];
	}

	cudaMemcpyToSymbol(config, host_leader, sizeof(float)* DIMENSION);

	cudaEventRecord(startEvent, 0);
	init_rand << <PRT_BLOCKS, THREADS >> >(dev_states, DIMENSION);
	make_rand << <PRT_BLOCKS, THREADS >> >(dev_states, dev_prt_vector, DIMENSION);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Random: %1.3f\n", elapsedTime);

	printf("Maximum na indexu %d\n", host_leader_index[0]);
	
	cudaFree(device_in);
	cudaFree(device_out);
	cudaFree(device_leader_index);
	cudaFree(device_maximum);
	cudaFree(dev_prt_vector);
	cudaFree(dev_states);
	
	cudaFreeHost(host_in);
	cudaFreeHost(host_out);
	
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	free(host_leader);
	free(host_leader_index);
	fclose(pFile);
}

