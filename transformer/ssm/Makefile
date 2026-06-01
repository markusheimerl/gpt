CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -lcudart -lcublasLt

test.out: ssm.o test.o
	$(CC) ssm.o test.o $(CUDALIBS) $(LDFLAGS) -o $@

ssm.o: ssm.c ssm.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ssm.c -o $@

test.o: test.c ssm.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c test.c -o $@

run: test.out
	@time ./test.out

clean:
	rm -f *.out *.o *.csv *.bin