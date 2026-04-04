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
APP_SOURCES := $(filter-out $(SRCDIR)/wasm_demo.cpp,$(SOURCES))
CORE_SOURCES := $(filter-out $(SRCDIR)/multi_layer_perceptron_main.cpp,$(APP_SOURCES))
NATIVE_SOURCES := $(APP_SOURCES)
GPU_OPENGL_LIBS ?= -lglfw -lGLEW -lGL
OPENCL_LIBS ?= -lOpenCL
WASM_EXPORTS := -s EXPORTED_FUNCTIONS='["_malloc","_free","_wasm_demo_reset","_wasm_demo_step","_wasm_demo_layer_count","_wasm_demo_layer_size","_wasm_demo_get_activation","_wasm_demo_get_weight","_wasm_demo_get_probability","_wasm_demo_get_label","_wasm_demo_get_prediction","_wasm_demo_get_loss","_wasm_demo_get_step_count","_wasm_demo_configure","_wasm_demo_clear_datasets","_wasm_demo_add_training_sample","_wasm_demo_add_validation_sample","_wasm_demo_select_validation_sample","_wasm_demo_get_training_sample_count","_wasm_demo_get_validation_sample_count","_wasm_demo_get_completed_epochs","_wasm_demo_get_training_progress","_wasm_demo_get_current_sample_index","_wasm_demo_get_current_validation_index","_wasm_demo_get_validation_pixel","_wasm_demo_get_validation_label","_wasm_demo_get_class_count"]'
WASM_RUNTIME := -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAP8","HEAPU8","HEAP32","HEAPF64"]'

REQUISITES:
	sudo apt-get install libglew-dev
	sudo apt-get -y install libglfw3
	sudo apt-get -y install libglfw3-dev
	sudo apt-get -y install ocl-icd-opencl-dev

requisites-opencl:
	sudo apt-get -y install ocl-icd-opencl-dev
	sudo apt-get -y install opencl-headers
	sudo apt-get -y install clinfo

clean:
	#rm ${PROGRAM}
	#rm letter-recognition.data

build: clean
	@mkdir -p $(BUILDDIR)
	# @echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<
	g++ -O3 -o $(PROGRAM) -std=c++17 $(INC) $(NATIVE_SOURCES)

build-gpu: clean
	@mkdir -p $(BUILDDIR)
	g++ -O3 -DMLP_WITH_OPENGL_COMPUTE -o $(PROGRAM) -std=c++17 $(INC) $(NATIVE_SOURCES) $(GPU_OPENGL_LIBS)

build-opencl: clean
	@mkdir -p $(BUILDDIR)
	g++ -O3 -DMLP_WITH_OPENCL -o $(PROGRAM) -std=c++17 $(INC) $(NATIVE_SOURCES) $(OPENCL_LIBS)

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
	./mlp mnist

run-letter-data-gpu: build-gpu
	./mlp letter-data --backend gpu

run-mnist-data-gpu: build-gpu
	./mlp mnist --backend gpu

run-letter-data-opencl: build-opencl
	./mlp letter-data --backend opencl

run-mnist-data-opencl: build-opencl
	./mlp mnist --backend opencl

prepare-zaland-mnist-data:
	mkdir -p data 
	wget -P data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
	wget -P data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
	wget -P data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
	wget -P data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
	cd data && gunzip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz

run-zalando-mnist-data: build
	./mlp mnist

serve-files:
	python3 -m http.server

opencl-diagnose:
	@echo "=== OpenCL ICD files ==="
	@ls -l /etc/OpenCL/vendors || true
	@echo
	@echo "=== NVIDIA ICD contents ==="
	@cat /etc/OpenCL/vendors/nvidia.icd || true
	@echo
	@echo "=== OpenCL loader ==="
	@ldconfig -p | grep OpenCL || true
	@echo
	@echo "=== NVIDIA OpenCL library deps ==="
	@ldd /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1 || true
	@echo
	@echo "=== WSL GPU libs ==="
	@ls -l /usr/lib/wsl/lib || true
	@echo
	@echo "=== NVIDIA SMI ==="
	@nvidia-smi || true
	@echo
	@echo "=== Environment ==="
	@echo "LD_LIBRARY_PATH=$$LD_LIBRARY_PATH"
	@echo
	@echo "=== clinfo ==="
	@clinfo || true
