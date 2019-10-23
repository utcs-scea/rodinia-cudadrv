include ../util/make.mk

#C compiler
CC = g++

SRC = clutils.cpp gaussianElim.cpp utils.cpp ../util/timing.c

CC_FLAGS = -g -O3

EXE = gaussian.out

release:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) \
		-L$(OPENCL_LIB) -l$(LIB_NAME) \
		-Wl,-rpath,$(OPENCL_LIB) \
		-I../util -DTIMING -Wno-unused-result

clean:
	rm -f $(EXE)
