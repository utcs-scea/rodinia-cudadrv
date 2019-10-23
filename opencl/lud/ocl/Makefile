include ../../util/make.mk
INC = -I../common
LIB = -L../$(OPENCL_LIB)

#C compiler
CC = g++
CC_FLAGS = -g -O3 -Wall

EXE = lud.out

$(EXE): lud.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) \
		-o $(EXE) \
		lud.cpp ../common/common.c ../../util/timing.c \
		$(INC) $(LIB) -l$(LIB_NAME) \
		-I../../util -DTIMING \
		-Wno-unused-result \
		-Wl,-rpath,ocl/../$(OPENCL_LIB) \
		-ldl

clean:
	rm -f ../$(EXE)
