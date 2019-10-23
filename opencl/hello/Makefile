NVIDIA_OCL_DIR=/usr/local/cuda
INTEL_OCL_DIR=/opt/intel/opencl
BUILD_DIR = ../../../../nwcc/cl_nw

all: nvidia intel overload

nvidia:
	gcc -I$(NVIDIA_OCL_DIR)/include/ -L$(NVIDIA_OCL_DIR)/lib64/ -o hello.gpu hello.c -lOpenCL

intel:
	gcc -I$(INTEL_OCL_DIR)/include/ -L$(INTEL_OCL_DIR) -o hello.cpu hello.c -lOpenCL

overload:
	gcc -I$(NVIDIA_OCL_DIR)/include/ -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -o hello.run hello.c -lguestlib -ldl

clean:
	rm *.cpu *.gpu *.run || true
