PROGRAM=mlp
CC := g++
SRCDIR := src
BUILDDIR := build
DISTDIR := dist
CFLAGS := -g # -Wall
INC := -I include
LIB :=
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
CORE_SOURCES := $(filter-out $(SRCDIR)/multi_layer_perceptron_main.cpp $(SRCDIR)/wasm_demo.cpp,$(SOURCES))
NATIVE_SOURCES := $(filter-out $(SRCDIR)/wasm_demo.cpp,$(SOURCES))
WASM_EXPORTS := -s EXPORTED_FUNCTIONS='["_malloc","_free","_wasm_demo_reset","_wasm_demo_step","_wasm_demo_layer_count","_wasm_demo_layer_size","_wasm_demo_get_activation","_wasm_demo_get_weight","_wasm_demo_get_probability","_wasm_demo_get_label","_wasm_demo_get_prediction","_wasm_demo_get_loss","_wasm_demo_get_step_count","_wasm_demo_configure","_wasm_demo_clear_datasets","_wasm_demo_add_training_sample","_wasm_demo_add_validation_sample","_wasm_demo_select_validation_sample","_wasm_demo_get_training_sample_count","_wasm_demo_get_validation_sample_count","_wasm_demo_get_completed_epochs","_wasm_demo_get_training_progress","_wasm_demo_get_current_sample_index","_wasm_demo_get_current_validation_index","_wasm_demo_get_validation_pixel","_wasm_demo_get_validation_label","_wasm_demo_get_class_count"]'
WASM_RUNTIME := -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAP8","HEAPU8","HEAP32","HEAPF64"]'

clean:
	#rm ${PROGRAM}
	#rm letter-recognition.data

build: clean
	@mkdir -p $(BUILDDIR)
	# @echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<
	g++ -O3 -o $(PROGRAM) -std=c++17 $(INC) $(NATIVE_SOURCES)

wasm-demo: clean
	@mkdir -p $(DISTDIR)
	em++ -O3 -std=c++17 $(INC) $(CORE_SOURCES) $(SRCDIR)/wasm_demo.cpp \
		-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 \
		-s MODULARIZE=1 -s EXPORT_ES6=1 \
		$(WASM_EXPORTS) $(WASM_RUNTIME) \
		-o $(DISTDIR)/mlp_demo.js
	cp web/index.html $(DISTDIR)/index.html
	cp web/mnist.html $(DISTDIR)/mnist.html

debug: clean
	g++ -O3 -D_GLIBCXX_DEBUG -o ${PROGRAM} -std=c++17 \
		$(INC) $(NATIVE_SOURCES)	
	./mlp letter-data

run-letter-data: build
	#wget https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data 
	./mlp letter-data

run-mnist-data: build
	#wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	#wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	#wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	#wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	#gunzip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
	./mlp mnist

run-zalando-mnist-data: build
	#wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
	#wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
	#wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
	#wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
	#gunzip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
	./mlp mnist
