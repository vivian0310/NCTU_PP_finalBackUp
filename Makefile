all:
	nvcc MLP_Network.cpp  MLP_HiddenLayer.cu MLP_OutputLayer.cpp MNIST.cpp main.cpp -o mlp
	#g++ -fopenmp MLP_Network.cpp  MLP_HiddenLayer.cpp MLP_OutputLayer.cpp MNIST.cpp main.cpp -o mlp
clean:
	rm -f mlp
