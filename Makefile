CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -Wno-unknown-cuda-version
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-path=/usr/lib/cuda --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -L/usr/lib/x86_64-linux-gnu -lcudart -lcublasLt

data.out: data.c
	$(CC) $(CFLAGS) data.c -o $@

data: data.out
	@./data.out

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

train: train.out
	@time ./train.out corpus.bin

cont: train.out
	@time ./train.out corpus.bin $$(ls -t *_gpt.bin 2>/dev/null | head -n1)

infer.out: infer.c
	$(CC) $(CFLAGS) infer.c -lopenblas -lm -o infer.out

infer: infer.out
	@./infer.out $$(ls -t *_gpt.bin 2>/dev/null | head -n1)

# Usage: make play <file.mid>
# e.g.   make play out.mid
#        make play sample_e00_c000.mid
PLAY_ARGS := $(filter-out play,$(MAKECMDGOALS))
play:
	@if [ -z "$(PLAY_ARGS)" ]; then \
		echo "usage: make play <file.mid>"; exit 1; \
	fi; \
	timidity "$(PLAY_ARGS)"
# Swallow the filename arg so make doesn't try to build it as a target.
%:
	@:

clean:
	rm -f *.out *.o *.csv *.mid
	$(MAKE) -C transformer/ clean
	$(MAKE) -C transformer/attention/ clean
	$(MAKE) -C transformer/mlp/ clean