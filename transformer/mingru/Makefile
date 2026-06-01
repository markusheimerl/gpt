CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -lcudart -lcublasLt

test.out: mingru.o test.o
	$(CC) mingru.o test.o $(CUDALIBS) $(LDFLAGS) -o $@

mingru.o: mingru.c mingru.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c mingru.c -o $@

test.o: test.c mingru.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c test.c -o $@

run: test.out
	@time ./test.out

clean:
	rm -f *.out *.o *.csv *.bin
