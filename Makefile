PROGRAM=mlp

clean:
	#rm ${PROGRAM}
	#rm letter-recognition.data

build: clean
	g++ -O3 -o ${PROGRAM} -std=c++14 multi_layer_perceptron_main.cpp \
		letters_data_reader.cpp \
		multi_layer_perceptron.cpp

debug: clean
	g++ -D_GLIBCXX_DEBUG -o ${PROGRAM} -std=c++14 multi_layer_perceptron_main.cpp \
		letters_data_reader.cpp \
		multi_layer_perceptron.cpp

run-letter-data: build
	#wget https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data 
	./mlp letter-data
