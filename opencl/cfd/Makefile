include ../util/make.mk

#C compiler
CC = g++

FLAGS = -O3
SRC = euler3d.cpp ../util/timing.c

EXE = euler3d.out

release:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) \
		-o $(EXE) \
		-I$(OPENCL_INC) -L$(OPENCL_LIB) -l$(LIB_NAME) \
		-I../util -DTIMING $(FLAGS) \
		-Wl,-rpath,$(OPENCL_LIB) \
		-ldl

errmsg:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D ERRMSG $(FLAGS)

ptx:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB)  -lOpenCL -D PTX_MSG $(FLAGS)

res:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB)  -lOpenCL -D RES_MSG $(FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
