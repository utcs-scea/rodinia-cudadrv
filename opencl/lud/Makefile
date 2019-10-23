all: lud

lud:
	cd ocl; make
	mv ocl/lud.out .;

clean:
	cd ocl; make clean

run: lud
	./lud.out -s 8192 -p 0 -d 0
