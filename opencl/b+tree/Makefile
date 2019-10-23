include ../util/make.mk

C_C = gcc
OCL_LIB = -l$(LIB_NAME)

b+tree.out:	./main.o \
		./kernel/kernel_gpu_opencl_wrapper.o \
		./kernel/kernel_gpu_opencl_wrapper_2.o \
		./util/timer/timer.o \
		./util/num/num.o \
		./util/opencl/opencl.o \
        ./timing.o
	$(C_C)	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./kernel/kernel_gpu_opencl_wrapper_2.o \
			./util/timer/timer.o \
			./util/num/num.o \
			./util/opencl/opencl.o \
            ./timing.o \
			-lm \
			-L$(OPENCL_LIB) \
			-Wl,-rpath,$(OPENCL_LIB) \
			$(OCL_LIB) \
			-ldl \
	        -o b+tree.out

run: b+tree.out
	./b+tree.out file data/mil.txt command data/command.txt -p 0 -d 0

main.o:	./common.h \
		./main.h \
		./main.c
	$(C_C)	$(KERNEL_DIM) ./main.c \
			-c \
			-I$(OPENCL_INC) \
			-o ./main.o \
			-Wno-unused-result \
			-DTIMING \
			-O3

./kernel/kernel_gpu_opencl_wrapper.o:	./common.h \
										./kernel/kernel_gpu_opencl_wrapper.h \
										./kernel/kernel_gpu_opencl_wrapper.c
	$(C_C)	$(KERNEL_DIM) ./kernel/kernel_gpu_opencl_wrapper.c \
			-c \
			-o ./kernel/kernel_gpu_opencl_wrapper.o \
			-O3 \
			-I../util \
			-DTIMING \
			-I$(OPENCL_INC)

./kernel/kernel_gpu_opencl_wrapper_2.o:	./common.h \
										./kernel/kernel_gpu_opencl_wrapper_2.h \
										./kernel/kernel_gpu_opencl_wrapper_2.c
	$(C_C)	$(KERNEL_DIM) ./kernel/kernel_gpu_opencl_wrapper_2.c \
			-c \
			-o ./kernel/kernel_gpu_opencl_wrapper_2.o \
			-O3 \
			-I../util \
			-DTIMING \
			-I$(OPENCL_INC)

timing.o: ../util/timing.h ../util/timing.c
	$(C_C) ../util/timing.c -I../util -c -o timing.o -I$(OPENCL_INC)

./util/timer/timer.o:	./common.h \
						./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)	./util/timer/timer.c \
			-c \
			-o ./util/timer/timer.o \
			-O3

./util/num/num.o:	./common.h \
					./util/num/num.h \
					./util/num/num.c
	$(C_C)	./util/num/num.c \
			-c \
			-o ./util/num/num.o \
			-O3

./util/opencl/opencl.o:	./common.h \
						./util/opencl/opencl.h \
						./util/opencl/opencl.c
	$(C_C)	./util/opencl/opencl.c \
			-c \
			-o ./util/opencl/opencl.o \
			-O3 \
			-I$(OPENCL_INC)

clean:
	rm	*.o *.out \
		./kernel/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		./util/opencl/*.o \
                output.txt 2>/dev/null || true
