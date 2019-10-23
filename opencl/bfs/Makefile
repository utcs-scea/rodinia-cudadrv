include ../util/make.mk

#C compiler
CC = g++

SRC = bfs.cpp timer.cc ../util/timing.c

CC_FLAGS = -g -O3

EXE = bfs.out

release:$(SRC)
	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) \
		-L$(OPENCL_LIB) -l$(LIB_NAME) \
		-Wl,-rpath,$(OPENCL_LIB) \
		-Wno-unused-result \
		-I../util -DTIMING \
		-ldl

errmsg:$(SRC)
	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) -L$(OPENCL_LIB) -lOpenCL -D ERRMSG PTX_MSG

ptx:$(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -L$(OPENCL_LIB) -lOpenCL -D PTX_MSG

profile:$(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -L$(OPENCL_LIB) -lOpenCL -D PROFILING

res:$(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -L$(OPENCL_LIB) -lOpenCL -D RES_MSG

debug: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -L$(OPENCL_LIB) -lOpenCL 

run:$(EXE)
	./$(EXE) graph4096.txt -p 0 -d 0

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
