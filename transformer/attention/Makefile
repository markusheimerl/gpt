CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -Wno-unknown-cuda-version
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-path=/usr/lib/cuda --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -L/usr/lib/x86_64-linux-gnu -lcudart -lcublasLt

test.out: attention.o test.o
	$(CC) attention.o test.o $(CUDALIBS) $(LDFLAGS) -o $@

attention.o: attention.c attention.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c attention.c -o $@

test.o: test.c attention.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c test.c -o $@

run: test.out
	@time ./test.out

clean:
	rm -f *.out *.o *.csv *.bin