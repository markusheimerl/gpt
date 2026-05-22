CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -Wno-unknown-cuda-version
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-path=/usr/lib/cuda --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -L/usr/lib/x86_64-linux-gnu -lcudart -lcublasLt

test.out: mlp.o test.o
	$(CC) mlp.o test.o $(CUDALIBS) $(LDFLAGS) -o $@

mlp.o: mlp.c mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c mlp.c -o $@

test.o: test.c mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c test.c -o $@

run: test.out
	@time ./test.out

clean:
	rm -f *.out *.o *.csv *.bin