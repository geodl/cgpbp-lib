/*
    Author: Johnathan M Melo Neto (jmmn.mg@gmail.com)

    This file is an adapted version of CGP-Library and Fast Artificial Neural Network Library (FANN)

    The original CGP-Library is available in <http://www.cgplibrary.co.uk>  
    The original FANN Library is available in <http://leenissen.dk/fann/wp/>

    Copyright (c) Andrew James Turner 2014, 2015 (andrew.turner@york.ac.uk)
    Copyright (c) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "cgpbplib.h"


double accuracy(struct parameters *, struct chromosome *, struct dataSet *);

int main(void)
{    

    struct parameters *params = NULL;
    struct dataSet *mainData = NULL;

    // Insert the desired dataset here
    mainData = initialiseDataSetFromFile("./dataSets/iris.txt");

    // Initialize general parameters
    int numInputs = 4;  // attributes
    int numOutputs = 3; // classes

    // Percentage of the sample size utilized: 0 < percentage <= 1
    double percentage = 1.00;   

    int numNodes = 500;
    int nodeArity = 20;
    double weightRange = 5;
    double mutationRate = 0.05;

    // Set general parameters
    params = initialiseParameters(numInputs, numNodes, numOutputs, nodeArity);
    addNodeFunction(params, "sig");
    setMutationType(params, "probabilistic");
    setConnectionWeightRange(params, weightRange);
    setMutationRate(params, mutationRate);
    
    fann_disable_seed_rand();

    // CGPBP-IN specific parameters
    int numGens_IN = 64;

    // CGPBP-OUT specific parameters
    int numGens_OUT = 40000;

    // Indexes to start the experiments
    int i_begin;
    int j_begin;
    int run_both;

    // Try to read files
    FILE *f_IN = fopen("./results/cgpbp_in.txt", "r+");
    FILE *f_OUT = fopen("./results/cgpbp_out.txt", "r+");

    if (f_IN == NULL || f_OUT == NULL ) // files do NOT exist
    {        
        // Create empty text files to store the results
        f_IN = fopen("./results/cgpbp_in.txt", "w");
        f_OUT = fopen("./results/cgpbp_out.txt", "w");

        // Header of the text files (to track the result of each independent run)
        fprintf(f_IN, "i,\tj,\tacc,\tmse\n");
        fprintf(f_OUT, "i,\tj,\tacc,\tmse\n");
        fflush(f_IN);
        fflush(f_OUT); 

        // Initialize experiments from the beginning
        i_begin = 0;
        j_begin = 0;
        run_both = 1;
        
    }
    else // files exist, so CONTINUE the ongoing experiment
    {
        fseek(f_IN, 0, SEEK_END);
        fseek(f_OUT, 0, SEEK_END);

        printf("Insert values to continue the experiments:\n");
    
        char buf[10];
        printf("i = ");
        if(!fgets(buf, sizeof(buf), stdin))
            exit(1);
        buf[strlen(buf) - 1] = 0;
        i_begin = strtol(buf, (char **)NULL, 10);
        printf("\n");
        printf("j = ");
        if(!fgets(buf, sizeof(buf), stdin))
            exit(1);
        buf[strlen(buf) - 1] = 0;
        j_begin = strtol(buf, (char **)NULL, 10);
        printf("\n");
        printf("run_both = ");
        if(!fgets(buf, sizeof(buf), stdin))
            exit(1);
        buf[strlen(buf) - 1] = 0;
        run_both = strtol(buf, (char **)NULL, 10);
        printf("\n");
    }
    
    // Header of the experiment output
    printf("TYPE\t\ti\tj\tacc\tmse\n\n");

    int i, j;
    
    for(i = i_begin; i < 3; i++) // 3 independent cross-validations
    {

        // Set seed (for reproducibility purpose)
        unsigned int seed = i + 50;
        shuffleData(mainData, &seed);
        struct dataSet * reducedData = reduceSampleSize(mainData, percentage);        
        struct dataSet ** folds = generateFolds(reducedData);

        for(j = j_begin; j < 10; j++) // stratified 10-fold cross-validation
        {  

            // CGP: Build training, validation, and testing sets
            int * training_index = (int*)malloc(7*sizeof(int));
            int * validation_index = (int*)malloc(2*sizeof(int));
            int testing_index = j;
            getIndex(training_index, validation_index, testing_index, &seed);

            struct dataSet *trainingData = getTrainingData(folds, training_index);    
            struct dataSet *validationData = getValidationData(folds, validation_index);
            struct dataSet *testingData = getTestingData(folds, testing_index);

            // FANN: Build training and validation sets
            struct fann_train_data * trainingDataFann = fann_read_train_from_cgp_format(trainingData->numSamples, trainingData->numInputs, trainingData->numOutputs, trainingData->inputData, trainingData->outputData);
            struct fann_train_data * validationDataFann = fann_read_train_from_cgp_format(validationData->numSamples, validationData->numInputs, validationData->numOutputs, validationData->inputData, validationData->outputData);

            // Training + Validation
            struct fann_train_data * completeTrainingDataFann = fann_merge_train_data(trainingDataFann, validationDataFann);

            // Set different seed for each independent run (for reproducibility purpose)
            unsigned int seed = (i*10)+j+5;

            if(run_both == 1)
            {
                // Run CGPBP-IN ***********************
                unsigned int bp_max_epochs_in = 4000;
                setBpMaxEpochs(params, bp_max_epochs_in);

                struct chromosome * bestChromoIN = runCGPBP_IN(params, trainingData, validationData, trainingDataFann, numGens_IN, &seed);
                setChromosomeFitness(params, bestChromoIN, testingData);
                double testMSE_IN = getChromosomeFitness(bestChromoIN);
                double testAccuracy_IN = accuracy(params, bestChromoIN, testingData);            
                freeChromosome(bestChromoIN);

                // Save the results
                fprintf(f_IN, "%d,\t%d,\t%.4lf,\t%.4lf\n", i, j, -testAccuracy_IN, testMSE_IN);
                
                fflush(f_IN);

                // Display the results
                printf("\nCGPBP-IN\t%d\t%d\t%.4lf\t%.4lf\n\n", i, j, -testAccuracy_IN, testMSE_IN);                 
            }
            
            // Run both methods from now on
            run_both = 1;                     

            // Run CGPBP-OUT ***********************
            unsigned int bp_max_epochs_out = 40000;
            setBpMaxEpochs(params, bp_max_epochs_out);

            struct chromosome * bestChromoOUT = runCGPBP_OUT(params, trainingData, validationData, completeTrainingDataFann, numGens_OUT, &seed);
            setChromosomeFitness(params, bestChromoOUT, testingData);
            double testMSE_OUT = getChromosomeFitness(bestChromoOUT); 
            double testAccuracy_OUT = accuracy(params, bestChromoOUT, testingData); 
            freeChromosome(bestChromoOUT);            

            // Save the results
            fprintf(f_OUT, "%d,\t%d,\t%.4lf,\t%.4lf\n", i, j, -testAccuracy_OUT, testMSE_OUT);
            
            fflush(f_OUT);

            // Display the results
            printf("\nCGPBP-OUT\t%d\t%d\t%.4lf\t%.4lf\n\n", i, j, -testAccuracy_OUT, testMSE_OUT);            

            // Clear training, validation, and testing sets
            freeDataSet(trainingData);
            freeDataSet(validationData);
            freeDataSet(testingData);

            free(training_index);
            free(validation_index);

            fann_destroy_train(trainingDataFann);
            fann_destroy_train(validationDataFann);
            fann_destroy_train(completeTrainingDataFann);
        }

        // Adjust start index of inner loop accordingly
        j_begin = 0;

        // Clear folds and reducedData
        int k;
        for(k = 0; k < 10; k++)
        {
            freeDataSet(folds[k]);
        }
        free(folds);        
        
        if(reducedData != mainData)
        {
            freeDataSet(reducedData);
        }   
    }
	
    // Free the remaining variables
    freeDataSet(mainData); 
    freeParameters(params);
    fclose(f_IN);
    fclose(f_OUT);

    printf("\n* * * * * END * * * * *\n"); 

    return 0;
}

/* 
    Accuracy: the proportion of correctly classified instances
    The output node that presents the higher value is defined as the class of the instance
    e.g. Consider a chromosome with 3 output nodes, their final values are:
    Output1: 0.25 | Output2: 0.34 | Output3: 0.09
    As Output2 presents the larger value, the instance is labeled as Class #2
    
    Here, we aim to minimize -(accuracy), which is equivalent to maximize +(accuracy)
*/
double accuracy(struct parameters *params, struct chromosome *chromo, struct dataSet *data)
{
    int i,j;
    int accuracy = 0;

    if(getNumChromosomeInputs(chromo) != getNumDataSetInputs(data))
    {
        printf("Error: the number of chromosome inputs must match the number of inputs specified in the dataSet.\n");
        printf("Terminating.\n");
        exit(0);
    }

    if(getNumChromosomeOutputs(chromo) != getNumDataSetOutputs(data))
    {
        printf("Error: the number of chromosome outputs must match the number of outputs specified in the dataSet.\n");
        printf("Terminating.\n");
        exit(0);
    }

    for(i = 0; i < getNumDataSetSamples(data); i++)
    {
        executeChromosome(chromo, getDataSetSampleInputs(data, i));

        double max_predicted = -DBL_MAX;
        int predicted_class = 0;
        int correct_class = 0;

        for(j = 0; j < getNumChromosomeOutputs(chromo); j++)
        {
            double current_prediction = getChromosomeOutput(chromo,j);
            
            if(current_prediction > max_predicted)
            {
            	max_predicted = current_prediction;
            	predicted_class = j;
            }

            if(getDataSetSampleOutput(data,i,j) == 1.0)
            {
            	correct_class = j;
            }            
        }

        if(predicted_class == correct_class)
        {
        	accuracy++;
        }    
    }

    return -accuracy / (double)getNumDataSetSamples(data);
}