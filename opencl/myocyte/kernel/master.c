// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"									// (in path provided here)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_fin.c"									// (in path provided here)
#include "../util/opencl/opencl.h"						// (in path provided here)

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>										// (in path known to compiler)	needed by printf
#include <CL/cl.h>										// (in path provided to compiler)	needed by OpenCL types and functions

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

#ifdef TIMING
#include "timing.h"
extern float kernel_time, d2h_time, h2d_time;
extern struct timeval tv_kernel_start, tv_kernel_end, tv_h2d_start, tv_h2d_end, tv_d2h_end;
extern struct timeval tv;
extern struct timeval tv_cpu_start, tv_cpu_end;
extern float time_cpu;
#endif

void 
master(	fp timeinst,
		fp *initvalu,
		fp *parameter,
		fp *finavalu,
		fp *com,

		cl_mem d_initvalu,
		cl_mem d_finavalu,
		cl_mem d_params,
		cl_mem d_com,

		cl_command_queue command_queue,
		cl_kernel kernel,

		long long *timecopyin,
		long long *timekernel,
		long long *timecopyout)
{

	//======================================================================================================================================================150
	//	VARIABLES
	//======================================================================================================================================================150

	// counters
	int i;

	// offset pointers
	int initvalu_offset_ecc;																// 46 points
	int initvalu_offset_Dyad;															// 15 points
	int initvalu_offset_SL;																// 15 points
	int initvalu_offset_Cyt;																// 15 poitns

	// common variables
	cl_int error;

#ifdef  TIMING
    gettimeofday(&tv_h2d_start, NULL);
#endif

	//======================================================================================================================================================150
	//	COPY DATA TO GPU MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	initvalu
	//====================================================================================================100

	int d_initvalu_mem;
	d_initvalu_mem = EQUATIONS * sizeof(fp);
	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									d_initvalu,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									d_initvalu_mem,			// size to be copied
									initvalu,				// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									NULL);   	// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	parameter
	//====================================================================================================100

	int d_params_mem;
	d_params_mem = PARAMETERS * sizeof(fp);
	error = clEnqueueWriteBuffer(	command_queue,
									d_params,
									1,
									0,
									d_params_mem,
									parameter,
									0,
									NULL,
									NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	END
	//====================================================================================================100

#ifdef  TIMING
    gettimeofday(&tv_kernel_start, NULL);
    tvsub(&tv_kernel_start, &tv_h2d_start, &tv);
    h2d_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	//======================================================================================================================================================150
	//	GPU: KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	KERNEL EXECUTION PARAMETERS
	//====================================================================================================100

	size_t local_work_size[1];
	local_work_size[0] = NUMBER_THREADS;
	size_t global_work_size[1];
	global_work_size[0] = 2*NUMBER_THREADS;

	// printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", (int)global_work_size[0]/(int)local_work_size[0], (int)local_work_size[0]);

	//====================================================================================================100
	//	KERNEL ARGUMENTS
	//====================================================================================================100

	clSetKernelArg(	kernel, 
					0, 
					sizeof(int), 
					(void *) &timeinst);
	clSetKernelArg(	kernel, 
					1, 
					sizeof(cl_mem), 
					(void *) &d_initvalu);
	clSetKernelArg(	kernel, 
					2, 
					sizeof(cl_mem), 
					(void *) &d_finavalu);
	clSetKernelArg(	kernel, 
					3, 
					sizeof(cl_mem), 
					(void *) &d_params);
	clSetKernelArg(	kernel, 
					4, 
					sizeof(cl_mem), 
					(void *) &d_com);

	//====================================================================================================100
	//	KERNEL
	//====================================================================================================100

	error = clEnqueueNDRangeKernel(	command_queue, 
									kernel, 
									1, 
									NULL, 
									global_work_size, 
									local_work_size, 
									0, 
									NULL, 
									NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Wait for all operations to finish, much like synchronizing threads in CUDA
	error = clFinish(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

#ifdef  TIMING
	gettimeofday(&tv_kernel_end, NULL);
	tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
	kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif


	//======================================================================================================================================================150
	//	COPY DATA TO SYSTEM MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	finavalu
	//====================================================================================================100

	int d_finavalu_mem;
	d_finavalu_mem = EQUATIONS * sizeof(fp);
	error = clEnqueueReadBuffer(command_queue,               // The command queue.
								d_finavalu,                  // The image on the device.
								CL_TRUE,                     // Blocking? (ie. Wait at this line until read has finished?)
								0,                           // Offset. None in this case.
								d_finavalu_mem, 			 // Size to copy.
								finavalu,                    // The pointer to the image on the host.
								0,                           // Number of events in wait list. Not used.
								NULL,                        // Event wait list. Not used.
								NULL);             // Event object for determining status. Not used.
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	com
	//====================================================================================================100

	int d_com_mem;
	d_com_mem = 3 * sizeof(fp);
	error = clEnqueueReadBuffer(command_queue,
								d_com,
								CL_TRUE,
								0,
								d_com_mem,
								com,
								0,
								NULL,
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	END
	//====================================================================================================100

#ifdef  TIMING
    gettimeofday(&tv_d2h_end, NULL);
    tvsub(&tv_d2h_end, &tv_kernel_end, &tv);
    d2h_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	//======================================================================================================================================================150
	//	CPU: FINAL KERNEL
	//======================================================================================================================================================150

	//======================================================================================================================================================150
	//	CPU: FINAL KERNEL
	//======================================================================================================================================================150

//    gettimeofday(&tv_cpu_start, NULL);

	initvalu_offset_ecc = 0;												// 46 points
	initvalu_offset_Dyad = 46;												// 15 points
	initvalu_offset_SL = 61;												// 15 points
	initvalu_offset_Cyt = 76;												// 15 poitns

	kernel_fin(	initvalu,
				initvalu_offset_ecc,
				initvalu_offset_Dyad,
				initvalu_offset_SL,
				initvalu_offset_Cyt,
				parameter,
				finavalu,
				com[0],
				com[1],
				com[2]);


//    gettimeofday(&tv_cpu_end, NULL);
//    tvsub(&tv_cpu_end, &tv_cpu_start, &tv);
//    time_cpu += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	//======================================================================================================================================================150
	//	COMPENSATION FOR NANs and INFs
	//======================================================================================================================================================150

	for(i=0; i<EQUATIONS; i++){
		if (isnan(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for NAN set rate of change to 0.0001
		}
		else if (isinf(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for INF set rate of change to 0.0001
		}
	}

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
