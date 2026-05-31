CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -lcudart -lcublasLt

train.out: gpt.o transformer/transformer.o transformer/ssm/ssm.o transformer/mlp/mlp.o train.o
	$(CC) gpt.o transformer/transformer.o transformer/ssm/ssm.o transformer/mlp/mlp.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

infer.out: infer.c
	$(CC) $(CFLAGS) infer.c $(LDFLAGS) -o $@

gpt.o: gpt.c gpt.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c gpt.c -o $@

transformer/transformer.o:
	$(MAKE) -C transformer transformer.o

transformer/ssm/ssm.o:
	$(MAKE) -C transformer/ssm ssm.o

transformer/mlp/mlp.o:
	$(MAKE) -C transformer/mlp mlp.o

train.o: train.c gpt.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

train: train.out
	@time ./train.out corpus.txt

run: train

infer: infer.out
	@time ./infer.out

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C transformer/ clean
