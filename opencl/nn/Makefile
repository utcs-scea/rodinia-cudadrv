include ../util/make.mk

#C compiler
CC = g++

SRC = nearestNeighbor.cpp clutils.cpp utils.cpp

CC_FLAGS = -g -O3

EXE = nn.out

release:
	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) \
		-I$(OPENCL_INC) -L$(OPENCL_LIB) -l$(LIB_NAME) \
		-I../util -DTIMING \
		-Wl,-rpath,$(OPENCL_LIB) \
		-ldl \
		-Wno-unused-result

clean:
	rm -f $(EXE)

run: release
	./$(EXE) filelist.txt -p 0 -d 0 -r 50 -lat 30 -lng 90 -q -t
