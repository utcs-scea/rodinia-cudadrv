include ../util/make.mk
LIBS = -l$(LIB_NAME)

ifndef CC
	CC = gcc
endif

OUTPUT = *.txt
CCFLAGS=-O3 -lm 

hotspot3D.out: 3D.c CL_helper.c ../util/timing.c
	$(CC) $^ $(CCFLAGS) -I$(OPENCL_INC) -L$(OPENCL_LIB) $(LIBS) \
		-I../util -DTIMING \
		-Wl,-rpath,$(OPENCL_LIB) \
		-ldl \
		-o $@

clean:
	rm -f hotspot3D.out $(OUTPUT) 
