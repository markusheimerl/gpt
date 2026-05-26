CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -lcudart -lcublasLt -lcudnn -lstdc++

test.out: transformer.o attention/attention.o attention/cudnn_att.o mlp/mlp.o test.o
	$(CC) transformer.o attention/attention.o attention/cudnn_att.o mlp/mlp.o test.o $(CUDALIBS) $(LDFLAGS) -o $@

transformer.o: transformer.c transformer.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c transformer.c -o $@

attention/attention.o:
	$(MAKE) -C attention attention.o

attention/cudnn_att.o:
	$(MAKE) -C attention cudnn_att.o

mlp/mlp.o:
	$(MAKE) -C mlp mlp.o

test.o: test.c transformer.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c test.c -o $@

run: test.out
	@time ./test.out

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C attention/ clean
	$(MAKE) -C mlp/ clean