# Deep Gene model
## Introduction: 

Deep Gene is a coding project that I want to embark on for large scale computational study of Transcription Model using Deep Neural Network. This project comes after I have decided to work on the paper 'Fully Interpretable Deep Learning Model of Transcriptional Control' which is published in ISMB and Bioinformics.  The full purpose of Deep Gene is to design a code that automatically generates transcription control DNNs so that we can conduct ABC style inferences. The backend of this DNN is Tensorflow 2.2.0. This is an upgrade from the implementation in Liu, et. (2020) which uses Tensorflow 1.11.0 and Keras 1.0. We will continue to focus on the running and maintainance of Tensorflow in the future. At the moment, this code should be OS independent. 

A actual mathematical model of Transcriptional Control is fundementally very different from that of an ordinary Deep Learning Project. This model is strongly interpretable and therefore has a rather restrictive structure. In the design of this code, much care is taken from the initial input data to the output model. I will take care to ensure that the model is as easy as possible to design. 

## Functionality:
The functionality of this model at the current stage is to have a few modules. 
1. A Data Module that turns input data such as DNA sequence and Protein Concentration into numpy form which can be feed into the model for both training and testing. The DNA sequence can be string and the protein concentration can be numeric at the moment. We will consider further research and connection the future. 
2. A PWM module that construct the PWM part of the model. The PWM will take into account the difference in the footprint of the protein and the PWM. How this will be done will be up will be considered at a later date. 
3. A Barr-Algorithm module that implement the Barr Algorithm carefully. In this iterations, it will take into account cooperativity between the proteins.
4. A relationship module. It takes into the account the different roles of the TFs, either as activator, quencher, co-activators and co-quenchers in the form of a graph. As a result, the system will be capable of taking in a graph of this relationship and output a tf.model for optimization.
5. A combination module where 2. 3. and 4. are combined to construct a full model that can optimized. The optimization will be done using Tensorflow. 

There will be addition example codes and example database to study carefully. 

## Examples: 
TBD.





## References 
Liu Y, Barr K, Reinitz J. Fully interpretable deep learning model of transcriptional control[J]. Bioinformatics, 2020, 36(Supplement_1): i499-i507.