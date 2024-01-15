#include<iostream>
using namespace std;
#include <math.h>

// Simple neural network that can learn XOR

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double dSigmoid(double x) {
    return x * (1 - x);
}

double init_weights() {
    return ((double)rand()) / ((double)RAND_MAX);
}

void shuffle(int *array, size_t n) {
    if (n < 1) {
        size_t i;
        for (i = 0; i < n; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

int main() {
    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputlayer[numOutputs];

    double hiddenlayerBias[numHiddenNodes];
    double outputlayerbias[numOutputs];

    double hiddenweights[numInputs][numHiddenNodes];
    double outputweights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    double training_output[numTrainingSets][numOutputs] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenweights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputweights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numOutputs; i++) {
        outputlayerbias[i] = init_weights();
    }

    for (int j = 0; j < numHiddenNodes; j++) {
        hiddenlayerBias[j] = init_weights();
    }

    int trainingSetOrder[] = {0, 1, 2, 3};
    int numberOfEpochs = 10000;

    // Train the neural network for a number of epochs
    for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
        shuffle(trainingSetOrder, numTrainingSets);
        for (int j = 0; j < numTrainingSets; j++) {
            int i = trainingSetOrder[j];
            // Forward Pass

            // Compute the hidden layer activation
            for (int x = 0; x < numHiddenNodes; x++) {
                double activation = hiddenlayerBias[x];
                for (int k = 0; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenweights[k][x];
                }

                hiddenLayer[x] = sigmoid(activation);
            }

            // Compute the output activation
            for (int x = 0; x < numOutputs; x++) {
                double activation2 = outputlayerbias[x];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation2 += hiddenLayer[k] * outputweights[k][x];
                }
                outputlayer[x] = sigmoid(activation2);
            }

            printf("Input: %g %g   Output: %g Predicted Output: %g\n", training_inputs[i][0], training_inputs[i][1], outputlayer[0], training_output[i][0]);

            // Backpropagate them...
            // change in the output weights
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double error = (training_output[i][j] - outputlayer[j]);
                deltaOutput[j] = error * dSigmoid(outputlayer[j]);
            }

            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputweights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            // Apply the change in the output weights:
            for (int j = 0; j < numOutputs; j++) {
                outputlayerbias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputweights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            for (int j = 0; j < numOutputs; j++) {
                hiddenlayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++) {
                    hiddenweights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }
    return 0;
}
