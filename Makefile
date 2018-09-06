PROGRAM=mlp
CC := g++
SRCDIR := src
BUILDDIR := build
CFLAGS := -g # -Wall
INC := -I include
LIB :=
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))

clean:
	#rm ${PROGRAM}
	#rm letter-recognition.data

build: clean
	@mkdir -p $(BUILDDIR)
	# @echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<
	g++ -O3 -o $(PROGRAM) -std=c++17 $(INC) $(SOURCES)

build-wasm: clean
	em++ -Os -std=c++17 $(INC) $(SOURCES) \
		--embed-file letter-recognition.data \
		-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 \
		-o b.html 
		#--shell-file html_template/shell_minimal.html \
		#--emrun -s WASM=1 -o a.html

debug: clean
	g++ -O3 -D_GLIBCXX_DEBUG -o ${PROGRAM} -std=c++17 \
		$(INC) $(SOURCES)	
	./mlp letter-data

run-letter-data: build
	#wget https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data 
	./mlp letter-data

run-mnist-data: build
	wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

