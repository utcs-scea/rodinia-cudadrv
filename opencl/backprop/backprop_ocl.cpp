// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "backprop.h"

#ifdef NV //NVIDIA
	#include <oclUtils.h>
#else 
	#include <CL/cl.h>
#endif

#ifdef TIMING
    #include "timing.h"
#endif

////////////////////////////////////////////////////////////////////////////////

// local variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_id   * device_list;
static cl_uint          num_devices;

// OCL config
int platform_id_inuse = 0;            // platform id in use (default: 0)
int device_id_inuse = 0;              //device id in use (default : 0)
cl_device_type device_type = CL_DEVICE_TYPE_GPU;

//Primitives for timing
#ifdef TIMING
struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, d2h_phy_time = 0, h2d_phy_time = 0;
#endif

static int initialize(void)
{
	cl_int result;
	size_t size;
    cl_uint num_platforms;

    // get OpenCL platforms
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(0,0,*) failed\n"); return -1; }
	cl_platform_id all_platform_id[num_platforms];
	if (clGetPlatformIDs(num_platforms, all_platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(*,*,0) failed\n"); return -1; }
    cl_platform_id platform_id = all_platform_id[platform_id_inuse];

    // get device
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices) != CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };
	printf("num_devices = %d\n", num_devices);
    if(device_id_inuse > num_devices) {
        printf("Invalid Device Number\n");
        return -1;
    }
	device_list = new cl_device_id[num_devices];
	//device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_list, NULL) != CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };

    // get device type
    if (clGetDeviceInfo(device_list[device_id_inuse], CL_DEVICE_TYPE, sizeof(device_type), (void *)&device_type, NULL)!= CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };

	// create OpenCL context
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU"); return -1; }

	// create command queue for the specific device
#ifdef TIMING
	cmd_queue = clCreateCommandQueue( context, device_list[device_id_inuse], CL_QUEUE_PROFILING_ENABLE, NULL );
#else
	cmd_queue = clCreateCommandQueue( context, device_list[device_id_inuse], 0, NULL );
#endif
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete[] device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = CL_DEVICE_TYPE_GPU;

	return 0;
}

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
	int in, hid, out;
	float out_err, hid_err;
  
	in = net->input_n;
	hid = net->hidden_n;
	out = net->output_n;   
   
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	// read the kernel core source
	const char * kernel_bp1  = "bpnn_layerforward_ocl";
	const char * kernel_bp2  = "bpnn_adjust_weights_ocl";
	const char * tempchar = "./backprop_kernel.cl";
	FILE * fp = fopen(tempchar, "rb"); 
	if(!fp) { printf("ERROR: unable to open '%s'\n", tempchar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

    probe_time_start(&ts_total);
    probe_time_start(&ts_init);

	if(initialize()) return -1;

	// compile kernel
	cl_int err = 0;
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	{ // show warnings/errors
		//static char log[65536]; memset(log, 0, sizeof(log));
		//cl_device_id device_id = 0;
		//err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
		//clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
		//if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}
	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return -1; }
    	
	cl_kernel kernel1;
	cl_kernel kernel2;
	kernel1 = clCreateKernel(prog, kernel_bp1, &err);  
	kernel2 = clCreateKernel(prog, kernel_bp2, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	clReleaseProgram(prog);

	init_time = probe_time_end(&ts_init);

	float *input_weights_one_dim;
    float *input_weights_prev_one_dim;
	float * partial_sum;
	float sum;
	float num_blocks = in / BLOCK_SIZE;
	
	input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
	input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
	partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
	
	// set global and local workitems
	size_t global_work[3] = { BLOCK_SIZE, BLOCK_SIZE * (size_t)num_blocks, 1 }; 
	size_t local_work[3] = { BLOCK_SIZE, BLOCK_SIZE, 1 };
	
	// this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
	// todo: fix mem allocation
	int m = 0;
	for (int k = 0; k <= in; k++) {	
		for (int j = 0; j <= hid; j++) {
		input_weights_one_dim[m] = net->input_weights[k][j];
		input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	    m++;
		}
	}

	cl_mem input_hidden_ocl;
	cl_mem input_ocl;
	cl_mem output_hidden_ocl;
	cl_mem hidden_partial_sum;
	cl_mem hidden_delta_ocl;
	cl_mem input_prev_weights_ocl;

    probe_time_start(&ts_memalloc);

	input_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_ocl\n"); return -1;}
	input_hidden_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_hidden_ocl\n"); return -1;}
	output_hidden_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer output_hidden_ocl\n"); return -1;}
	hidden_partial_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, num_blocks * WIDTH * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer hidden_partial_sum\n"); return -1;}
	hidden_delta_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer hidden_delta_ocl\n"); return -1;}
	input_prev_weights_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_prev_weights_ocl\n"); return -1;}
    printf("Memory buffer size: %lf MB\n", ((in + 1) * sizeof(float) + (in + 1) * (hid + 1) * sizeof(float) +
                (hid + 1) * sizeof(float) + num_blocks * WIDTH * sizeof(float) + (hid + 1) * sizeof(float) +
                (in + 1) * (hid + 1) * sizeof(float)) / 1024.0 / 1024);


    mem_alloc_time = probe_time_end(&ts_memalloc);

	printf("Performing %s computation\n", device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");
    cl_event event;

    probe_time_start(&ts_h2d);

	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, input_ocl, 1, 0, (in + 1) * sizeof(float), net->input_units, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_ocl\n"); return -1; }
    h2d_phy_time += probe_event_time(event,cmd_queue);
    clReleaseEvent(event);

	err = clEnqueueWriteBuffer(cmd_queue, input_hidden_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_hidden_ocl\n"); return -1; }
    h2d_phy_time += probe_event_time(event,cmd_queue);
    clReleaseEvent(event);

    h2d_time = probe_time_end(&ts_h2d);
    probe_time_start(&ts_kernel);

	clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &input_ocl);
	clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &output_hidden_ocl);
	clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &input_hidden_ocl);
	clSetKernelArg(kernel1, 3, sizeof(void *), (void*) &hidden_partial_sum );
	clSetKernelArg(kernel1, 4, sizeof(float) *  HEIGHT, (void*)NULL );
	clSetKernelArg(kernel1, 5, sizeof(float ) *  HEIGHT * WIDTH, (void*)NULL );
	clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &in);
	clSetKernelArg(kernel1, 7, sizeof(cl_int), (void*) &hid);

	err = clEnqueueNDRangeKernel(cmd_queue, kernel1, 2, NULL, global_work, local_work, 0, 0, NULL);
	if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	

    clFinish(cmd_queue);
    kernel_time = probe_time_end(&ts_kernel);
    probe_time_start(&ts_d2h);

	err = clEnqueueReadBuffer(cmd_queue, hidden_partial_sum, 1, 0, num_blocks * WIDTH * sizeof(float), partial_sum, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueReadBuffer: partial sum\n"); return -1; }	
    d2h_phy_time += probe_event_time(event,cmd_queue);
    clReleaseEvent(event);

    d2h_time = probe_time_end(&ts_d2h);

	for (int j = 1; j <= hid; j++) {
		sum = 0.0;
		for (int k = 0; k < num_blocks; k++) {	
		sum += partial_sum[k * hid + j-1] ;
    }
		sum += net->input_weights[0][j];
		net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
	}


	bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
	bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
	bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
	bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

    probe_time_start(&ts_h2d);

	err = clEnqueueWriteBuffer(cmd_queue, hidden_delta_ocl,       1, 0, (hid + 1) * sizeof(float), net->hidden_delta, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer hidden_delta_ocl\n"); return -1; }
    h2d_phy_time += probe_event_time(event,cmd_queue);
    clReleaseEvent(event);

	err = clEnqueueWriteBuffer(cmd_queue, input_prev_weights_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_prev_one_dim, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_prev_weights_ocl\n"); return -1; }
    h2d_phy_time += probe_event_time(event,cmd_queue);
    clReleaseEvent(event);

	err = clEnqueueWriteBuffer(cmd_queue, input_hidden_ocl,       1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_hidden_ocl\n"); return -1; }
    h2d_phy_time += probe_event_time(event,cmd_queue);
    clReleaseEvent(event);

    h2d_time += probe_time_end(&ts_h2d);
    probe_time_start(&ts_kernel);

	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &hidden_delta_ocl);
	clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*) &hid);
	clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &input_ocl);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &in);
	clSetKernelArg(kernel2, 4, sizeof(void *), (void*) &input_hidden_ocl);
	clSetKernelArg(kernel2, 5, sizeof(void *), (void*) &input_prev_weights_ocl );

	err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 2, NULL, global_work, local_work, 0, 0, NULL);
	if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	

    clFinish(cmd_queue);
    kernel_time += probe_time_end(&ts_kernel);
    probe_time_start(&ts_d2h);

	err = clEnqueueReadBuffer(cmd_queue, input_ocl, 1, 0, (in + 1) * sizeof(float), net->input_units, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueReadBuffer: input_ocl\n"); return -1; }	
    d2h_phy_time += probe_event_time(event,cmd_queue);
    clReleaseEvent(event);

	err = clEnqueueReadBuffer(cmd_queue, input_hidden_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueReadBuffer: input_hidden_ocl\n"); return -1; }	
    d2h_phy_time += probe_event_time(event,cmd_queue);
    clReleaseEvent(event);

    d2h_time += probe_time_end(&ts_d2h);
    probe_time_start(&ts_close);

	clReleaseMemObject(input_ocl);
	clReleaseMemObject(output_hidden_ocl);
	clReleaseMemObject(input_hidden_ocl);
	clReleaseMemObject(hidden_partial_sum);
	clReleaseMemObject(input_prev_weights_ocl);
    shutdown();

	close_time = probe_time_end(&ts_close);

	free(input_weights_prev_one_dim);
	free(partial_sum);
	free(input_weights_one_dim);

	total_time = probe_time_end(&ts_total);

	printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("HtoD_Phy: %f\n", h2d_phy_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("DtoH_Phy: %f\n", d2h_phy_time);
	printf("Close: %f\n", close_time);
	printf("API: %f\n", init_time+mem_alloc_time+h2d_time+kernel_time+d2h_time+close_time);
	printf("Total: %f\n", total_time);
}
