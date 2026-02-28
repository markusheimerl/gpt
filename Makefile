CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -Wno-unknown-cuda-version
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -L/opt/cuda/lib64 -lcudart -lcublasLt

train.out: gpt.o transformer/transformer.o transformer/attention/attention.o transformer/mlp/mlp.o train.o
	$(CC) gpt.o transformer/transformer.o transformer/attention/attention.o transformer/mlp/mlp.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

gpt.o: gpt.c gpt.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c gpt.c -o $@

transformer/transformer.o:
	$(MAKE) -C transformer/ transformer.o

transformer/attention/attention.o:
	$(MAKE) -C transformer/attention/ attention.o

transformer/mlp/mlp.o:
	$(MAKE) -C transformer/mlp/ mlp.o

train.o: train.c gpt.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out corpus.txt

cont: train.out
	@time ./train.out corpus.txt $$(ls -t *_gpt.bin 2>/dev/null | head -n1)

clean:
	rm -f *.out *.o *.csv
	$(MAKE) -C transformer/ clean
	$(MAKE) -C transformer/attention/ clean
	$(MAKE) -C transformer/mlp/ clean