//
//  main.cpp
//  Neural Network Testing
//
//  Created by Johnny Chen on 12/9/15.
//  Copyright (c) 2015 JohnnyChen. All rights reserved.
//

#include <iostream>
#include <fstream> //reading from file
#include <vector>
#include <math.h> //to use the activation function because we need e
#include <iomanip> //for 3 decimal places

using namespace std;

string file1, file2, file3;

//from the first file
int inputNodes, hiddenNodes, outputNodes;
vector<vector<double>> weightsToHidden;
vector<vector<double>> weightsToOutput;
vector<vector<double>> network;

//from the 2nd file
int numTrainingExamples, inputs, outputs;
vector<vector<double>> exampleInputs;
vector<vector<double>> exampleOutputs; //0 or 1
vector<vector<double>> examples;

//user enters this
int epoch;
double learningRate;

void readFromFile1(string name);
void readFromFile2(string name);
void writeNetworkToFile(string name, vector<vector<double>> network);

vector<vector<double>> backPropLearning(vector<vector<double>> examples, vector<vector<double>> network);

int main(int argc, const char * argv[]) {
    
    cout << "Please enter the file containing the neural network\n";
    //read the file
    //    cin >> file1;
    file1 = "1neuralNetwork.txt";
    readFromFile1(file1);
    
    network.reserve( weightsToHidden.size() + weightsToOutput.size() ); // preallocate memory
    network.insert( network.end(), weightsToHidden.begin(), weightsToHidden.end() );
    network.insert( network.end(), weightsToOutput.begin(), weightsToOutput.end() );
    
    cout << "Please enter the file containing the test set.\n";
    //    cin >> file2;
    //    readFromFile2(file2);
    file2 = "1testingExamples.txt";
    readFromFile2(file2);
    
    examples.reserve( exampleInputs.size() + exampleOutputs.size() ); // preallocate memory
    examples.insert( examples.end(), exampleInputs.begin(), exampleInputs.end() );
    examples.insert( examples.end(), exampleOutputs.begin(), exampleOutputs.end() );
    
    cout << "Where would you like to output the results to?\n";
    file3 = "1compareToTestResults.txt";
    
    cout << "Choose epoch.\n";
    //    cin >> epoch;
    epoch = 1;
    
    cout << "Choose learning rate.\n";
    //    cin >> learningRate;
    learningRate = 0.1;
    
    vector<vector<double>> newNetwork = backPropLearning(examples, network);
    writeNetworkToFile(file3, newNetwork);
    
    return 0;
}

void readFromFile1(string name){
    /*
     1st line has 3 integers
     - number of input nodes
     - number of hidden nodes
     - number of output nodes
     
     Next Nh lines each has Ni + 1 weights entering the 1st hidden node, then 2nd ... until Nh
     There is a +1 because a bias weight which is always the first weight
     Weights are doubleing point numbers
     
     Next No has Nh + 1 weights
     */
    
    ifstream myFile;
    myFile.open(file1);
    if (myFile.is_open()){
        //Put contents of file into array
        for (int i = 0; i <= 3; i++){
            if (i == 0)
                myFile >> inputNodes;
            if (i == 1)
                myFile >> hiddenNodes;
            if (i == 2){
                myFile >> outputNodes;
            }
        }
        weightsToHidden.resize(hiddenNodes);
        for (int j = 0; j < hiddenNodes; j++){
            weightsToHidden[j].resize(inputNodes+1);
            for (int m = 0; m <= inputNodes; m++){
                //                cout << "j = " << j << "m = " << m << endl;
                myFile >> weightsToHidden[j][m];
            }
        }
        weightsToOutput.resize(outputNodes);
        for (int k = 0; k < outputNodes; k++){
            weightsToOutput[k].resize(hiddenNodes+1);
            for (int n = 0; n <= hiddenNodes; n++){
                myFile >> weightsToOutput[k][n];
            }
        }
        myFile.close();
    }
    else
        cout << "Unable to open file.\n";
}

//training examples to train
void readFromFile2(string name){
    /*
     1st line has 3 integers
     - number of training examples
     - number of input nodes(will match file1)
     - number of output nodes(will match file1)
     
     every other line is an example
     - Ni doubleing point inputs
     - No Boolean output(0 or 1)
     */
    
    ifstream myFile;
    myFile.open(file2);
    if (myFile.is_open()){
        //Put contents of file into array
        for (int i = 0; i <= 3; i++){
            if (i == 0)
                myFile >> numTrainingExamples;
            if (i == 1)
                myFile >> inputs;
            if (i == 2){
                myFile >> outputs;
            }
        }
        if (inputs != inputNodes || outputs != outputNodes){
            cout << "The input and output values do not match the initial neural network.";
        }
        else{
            int sum = inputs + outputs;
            
            exampleInputs.resize(numTrainingExamples);
            exampleOutputs.resize(numTrainingExamples);
            //go through each line one at a time, put the stuff in a vector, and do the backprop
            for (int i = 0; i < numTrainingExamples; i++){
                exampleInputs[i].resize(inputs);
                exampleOutputs[i].resize(outputs);
                for (int j = 0; j < sum; j++){
                    if (j < inputs){
                        myFile >> exampleInputs[i][j];
                    }
                    else {
                        myFile >> exampleOutputs[i][j-inputs];
                    }
                }
            }
            myFile.close();
        }
    }
    else
        cout << "Unable to open file.\n";
}

double applyActivFunct(double x){
    double result = 1/(1+exp(-1*x));
    return result;
}

double applyDerivActivFunct(double x){
    double result = applyActivFunct(x) * (1 - applyActivFunct(x));;
    return result;
}

//given return a neural network and the examples, return a neural network
//examples include both input and output examples. same with network. contains first weight from input to hidden node, then hidden node to output
vector<vector<double>> backPropLearning(vector<vector<double>> examples, vector<vector<double>> network){
    
    //error for both outlayer and hiddenlayer
    vector<vector<double>> errors;
    errors.resize(2);
    errors[0].resize(outputNodes);
    errors[1].resize(hiddenNodes);
    
    int loop = 0;
    
    while (loop < epoch){
        //go through each example in examples
        //you could do numTrainingExamples or you can do the exampleInputs.size()
        for (int i = 0; i < numTrainingExamples; i++){
            
            //base will contain all the example inputs
            vector<double> base = examples[i];
            
            //propagate the inputs forward to compute the outputs
            vector<double> middle;
            middle.resize(hiddenNodes);
            
            //loop through each hidden node
            for (int j = 0; j < hiddenNodes; j++){
                double result = 0;
                //get contribution from each node from previous layer
                for (int k = 0; k < inputNodes+1; k++){
                    //for the first weight, multiply by the fixed input -1
                    if (k == 0){
                        result += -1 * network[j][0];
                    }
                    else {
                        //network has the bias weight at the beginning and is ahead of the example/base
                        result += network[j][k] * base[k-1];
                    }
                }
                middle[j] = result;
            }
            
            //output
            vector<double> top;
            top.resize(outputNodes);
            for (int j = 0; j < outputNodes; j++){
                double result = 0;
                for (int k = 0; k < hiddenNodes+1; k++){
                    if (k == 0){
                        result += -1 * network[hiddenNodes+j][0];
                    }
                    else {
                        result += network[hiddenNodes+j][k] * applyActivFunct(middle[k-1]);
                    }
                }
                top[j] = result;
            }
            
            float A, B, C, D;
            
            for (int j = 0; j < outputNodes; j++){
                //the activation of output nodes should be rounded to 1 or 0
                double actualOutput = (applyActivFunct(top[j]) >= 0.5) ? 1 : 0;
            
                //compare the actual output with the expected output from the training set(examples)
                double expectedOutput = examples[numTrainingExamples+i][j];
                
                //based on the contingency table
                if (actualOutput == 1 && expectedOutput == 1){
                    A++;
                }
                else if (actualOutput == 1 && expectedOutput == 0){
                    B++;
                }
                else if (actualOutput == 0 && expectedOutput == 1){
                    C++;
                }
                else if (actualOutput == 0 && expectedOutput == 0){
                    D++;
                }
            }
            
            float overallAccuracy, precision, recall, f1;
            
            overallAccuracy = (A+D)/(A+B+C+D);
            precision = A/(A+B);
            recall = A/(A+C);
            f1 = (2*precision*recall)/(precision+recall); //will have a value in between the precision and recall, closer to the lower of the two
            
            
        }
        loop++;
    }
    
    return network;
}

void writeNetworkToFile(string name, vector<vector<double>> network){
    ofstream myfile;
    
    
    myfile.open (name);
    if (myfile.is_open())
    {
        myfile << inputNodes << " " << hiddenNodes << " " << outputNodes << "\n";
        
        //size is hidden + output
        for (int i = 0; i < hiddenNodes; i++){
            //            myfile << "This is another line.\n";
            for (int j = 0; j < inputNodes + 1; j++){
                myfile << fixed << setprecision(3) << network[i][j] << " ";
                //ceil(num*pow(10,x))/pow(10,x)
                
            }
            myfile << "\n";
        }
        for (int i = 0; i < outputNodes; i++){
            for (int j = 0; j < hiddenNodes + 1; j++){
                myfile << fixed << setprecision(3) << network[hiddenNodes+i][j] << " ";
            }
            myfile << "\n";
        }
        myfile.close();
    }
    else cout << "Unable to open file";
}



