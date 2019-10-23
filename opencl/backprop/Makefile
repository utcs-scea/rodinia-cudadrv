include ../util/make.mk

# C compiler
CC = g++
CC_FLAGS = -g -O2

backprop.out: backprop.o imagenet.o facetrain.o timing.o backprop_ocl.o
	$(CC) $(CC_FLAGS) \
		-o backprop.out \
		-L$(OPENCL_LIB) \
		-Wl,-rpath,$(OPENCL_LIB) \
		backprop.o imagenet.o facetrain.o timing.o backprop_ocl.o \
		-l$(LIB_NAME) \
		-ldl

%.o: %.c
	$(CC) $(CC_FLAGS) $< -c

backprop_ocl.o: backprop_ocl.cpp
	$(CC) $(CC_FLAGS) -I../util backprop_ocl.cpp -c -Wno-unused-result -Wno-deprecated-declarations -DTIMING

timing.o: ../util/timing.c
	$(CC) $(CC_FLAGS) -I../util ../util/timing.c -c

backprop.o: backprop.c
	$(CC) $(CC_FLAGS) backprop.c -c -Wno-unused-result

run: backprop.out
	./backprop.out -n 4194304 -p 0 -d 0

clean:
	rm -f *.o *~ *.out *.linkinfo
