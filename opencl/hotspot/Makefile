include ../util/make.mk

DATA_DIR = ~/nw/gpu-rodinia/data

CC = gcc
CC_FLAGS = -g -O3 -Wall

EXE = hotspot.out

$(EXE): hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $(EXE) \
		-I$(OPENCL_INC) OpenCL_helper_library.c hotspot.c ../util/timing.c \
		-L$(OPENCL_LIB) -l$(LIB_NAME) -I../util -DTIMING \
		-Wl,-rpath,$(OPENCL_LIB) \
		-ldl

run:
	./$(EXE) 1024 4 10000 $(DATA_DIR)/hotspot/temp_1024 $(DATA_DIR)/hotspot/power_1024 output.txt -p 0 -d 0

clean:
	rm -f $(EXE) *.txt || true
