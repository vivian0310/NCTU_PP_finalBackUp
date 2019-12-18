#include <omp.h>

#include "MLP_HiddenLayer.h"

__global__ void forwardParallel(int nCurrentUnit, int nPreviousUnit, float *inputLayer_D, float *outputLayer_D, float *weight_D, float *biasWeight_D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x ;
    if (j < nCurrentUnit)
    {
        float net= 0;
        for(int i = 0 ; i < nPreviousUnit ; i++)
        {
            net += inputLayer_D[i] * weight_D[j*nPreviousUnit+i];
        }
        net+=biasWeight_D[j];
        
        outputLayer_D[j] = 1.F/(1.F + (float)exp(-net));
    }
}
float* MLP_HiddenLayer::ForwardPropagate(float* inputLayers)      // f( sigma(weights * inputs) + bias )
{
    this->inputLayer=inputLayers;

    float *inputLayer_D, *outputLayer_D, *weight_D, *biasWeight_D;
    int size_F = sizeof(float);
    cudaMalloc(&inputLayer_D, 784 * size_F);
    cudaMalloc(&outputLayer_D, 512 * size_F);
    cudaMalloc(&weight_D, 410000 * size_F);
    cudaMalloc(&biasWeight_D, 512 * size_F);
    
    cudaMemcpy(inputLayer_D, inputLayer, 784 *size_F, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_D, weight, 410000 *size_F, cudaMemcpyHostToDevice);
    cudaMemcpy(biasWeight_D, biasWeight, 512 *size_F, cudaMemcpyHostToDevice);

    int threadsPerBlock = 64;
    int numBlocks = ((nCurrentUnit%threadsPerBlock) == 0)? \
                     (nCurrentUnit/threadsPerBlock) : (nCurrentUnit/threadsPerBlock) + 1;
    forwardParallel <<<numBlocks, threadsPerBlock>>> (nCurrentUnit, nPreviousUnit, inputLayer_D, outputLayer_D, weight_D, biasWeight_D);

    cudaMemcpy(outputLayer, outputLayer_D, 512 * size_F, cudaMemcpyDeviceToHost);
    // for(int j = 0 ; j < nCurrentUnit ; j++)
    // {
    //     float net= 0;
    //     for(int i = 0 ; i < nPreviousUnit ; i++)
    //     {
    //         net += inputLayer[i] * weight[j*nPreviousUnit+i];
    //     }
    //     net+=biasWeight[j];
        
    //     outputLayer[j] = ActivationFunction(net);
    // }
    cudaFree(inputLayer_D);
    cudaFree(outputLayer_D);
    cudaFree(biasWeight_D);
    cudaFree(weight_D);

    return outputLayer;
}

void MLP_HiddenLayer::BackwardPropagateHiddenLayer(MLP_OutputLayer* previousLayer)
{
    
    float* previousLayer_weight = previousLayer->GetWeight();
    float* previousLayer_delta = previousLayer->GetDelta();
    int previousLayer_node_num = previousLayer->GetNumCurrent();

    for (int j = 0; j < nCurrentUnit; j++)
    {
        float previous_sum=0;
        for (int k = 0; k < previousLayer_node_num; k++)
        {
            previous_sum += previousLayer_delta[k] * previousLayer_weight[k*nCurrentUnit + j];
        }
        delta[j] =  outputLayer[j] * (1 - outputLayer[j])* previous_sum;
        //delta[j] =  DerivativeActivation(j)* previous_sum;
    }
    
    for (int j = 0; j < nCurrentUnit; j++)
        for (int i = 0; i < nPreviousUnit ; i++)
            gradient[j*nPreviousUnit + i] +=  -delta[j] * inputLayer[i];
    
    for (int j = 0 ; j < nCurrentUnit   ; j++)
        biasGradient[j] += -delta[j] ;
}

void MLP_HiddenLayer::UpdateWeight(float learningRate)
{
    for (int j = 0; j < nCurrentUnit; j++)
        for (int i = 0; i < nPreviousUnit; i++)
            weight[j*nPreviousUnit + i] +=  -learningRate *gradient[j*nPreviousUnit + i];
    
    for (int j = 0; j < nCurrentUnit; j++)
        biasWeight[j] += -biasGradient[j];
    
    for (int j = 0; j < nCurrentUnit; j++)           
        for (int i = 0; i < nPreviousUnit; i++)
            gradient[j*nPreviousUnit + i] = 0;
    
    for (int j = 0; j < nCurrentUnit; j++)
        biasGradient[j]=0;
}


int MLP_HiddenLayer::GetMaxOutputIndex()
{
    int maxIdx = 0;
    for(int o = 1; o < nCurrentUnit; o++){
        if(outputLayer[o] > outputLayer[maxIdx])
            maxIdx = o;
    }
    
    return maxIdx;
}


