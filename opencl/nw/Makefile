include ../util/make.mk

#C compiler
CC = g++

CC_FLAGS = -g -O3 -Wall

EXE = nw.out

$(EXE): nw.c
	$(CC) ${KERNEL_DIM} $(CC_FLAGS) -o $(EXE) nw.c ../util/timing.c \
		-Wl,-rpath,$(OPENCL_LIB) \
		-I$(OPENCL_INC) -L$(OPENCL_LIB) -l$(LIB_NAME) \
		-I../util -DTIMING \
		-Wno-unused-result \
		-ldl

clean:
	rm -f $(EXE)
