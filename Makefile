PROGRAM=mlp

clean:
	#rm ${PROGRAM}
	#rm letter-recognition.data

build: clean
	g++ -O3 -o ${PROGRAM} -std=c++17 \
		multi_layer_perceptron_main.cpp \
		letters_data_reader.cpp \
		multi_layer_perceptron.cpp

build-wasm: clean
	em++ -Os -std=c++17 multi_layer_perceptron_main.cpp \
		letters_data_reader.cpp \
		multi_layer_perceptron.cpp \
		--embed-file letter-recognition.data \
		-s WASM=1 -s ALLOW_MEMORY_GROWTH=1\
		-o b.html 
		#--shell-file html_template/shell_minimal.html \
		#--emrun -s WASM=1 -o a.html

debug: clean
	g++ -D_GLIBCXX_DEBUG -o ${PROGRAM} -std=c++17 multi_layer_perceptron_main.cpp \
		letters_data_reader.cpp \
		multi_layer_perceptron.cpp
	./mlp letter-data

run-letter-data: build
	#wget https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data 
	./mlp letter-data

run-mnist-data: build
	wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

