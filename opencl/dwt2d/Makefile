#all:
#	g++ -o components.o -I/usr/local/cuda-5.5/include -c components.cpp
#	g++ -o prog  -I/usr/local/cuda-5.5/include main.cpp components.o -lOpenCL

include ../util/make.mk
OCL_LIB = -l$(LIB_NAME) -L$(OPENCL_LIB)

ifdef OUTPUT
	override OUTPUT = -DOUTPUT
endif

OUTPUT = -DOUTPUT

C_C = g++
OCL_INC = -I$(OPENCL_INC)

default:	dwt2d

components:
	$(C_C) -o components.o $(OCL_INC) -c components.cpp 

dwt2d:
	$(C_C) -o dwt2d.out $(OCL_INC) \
		-I../util -DTIMING \
		main.cpp ../util/timing.c \
		$(components) \
		$(OUTPUT) \
		$(OCL_LIB) \
		-Wl,-rpath,$(OPENCL_LIB) \
		-ldl

clean:
	rm -rf *.o dwt2d.out
	rm -rf *.bmp.dwt.*

