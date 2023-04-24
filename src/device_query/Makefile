INCLUDES := -I../third-parties/argparse/ -I.
CFILES := ../third-parties/argparse/argparse.c

all: install

install: device_query
	mkdir -p ~/.local/bin
	cp device_query ~/.local/bin

device_query: device_query.cu helper_cuda.h helper_string.h
	nvcc -w $(INCLUDES) -o device_query device_query.cu $(CFILES) -arch=compute_50
