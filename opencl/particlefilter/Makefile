include ../util/make.mk

# C compiler
CC = g++
CC_FLAGS = -O2 
INCLUDES = -I$(OPENCL_INC) -I../util

ALL: OCL_particlefilter_double

OCL_particlefilter_double: ex_particle_OCL_double_seq.o timing.o
	$(CC) $(CC_FLAGS) ex_particle_OCL_double_seq.o timing.o \
		$(INCLUDES)  -L$(OPENCL_LIB) -l$(LIB_NAME) \
		-Wl,-rpath,$(OPENCL_LIB) \
		-ldl \
		-o OCL_particlefilter_double.out

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

timing.o: ../util/timing.c
	$(CC) $(CC_FLAGS) $(INCLUDES) -I../util ../util/timing.c -c

ex_particle_OCL_double_seq.o: ex_particle_OCL_double_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_double_seq.cpp -DTIMING -c -Wno-unused-result

clean:
	rm -f *.o *~ OCL_particlefilter_*.out *.linkinfo
