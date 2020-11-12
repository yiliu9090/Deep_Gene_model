# Deep Gene model (Beta)
## Introduction: 

Deep Gene is a coding project that I want to embark on for large scale computational study of Transcription Model using Deep Neural Network. This project comes after I have decided to work on the paper 'Fully Interpretable Deep Learning Model of Transcriptional Control' which is published in ISMB and Bioinformics.  The full purpose of Deep Gene is to design a code that automatically generates transcription control DNNs so that we can conduct ABC style inferences. The backend of this DNN is Tensorflow 2.2.0 (Abadi (2015)).  This is an upgrade from the implementation in Liu, et. (2020) which uses Tensorflow 1.11.0 and Keras 1.0. We will continue to focus on the running and maintainance of Tensorflow in the future. At the moment, this code should be OS independent. 

A actual mathematical model of Transcriptional Control is fundementally very different from that of an ordinary Deep Learning Project. This model is strongly interpretable and therefore has a rather restrictive structure. In the design of this code, much care is taken from the initial input data to the output model. I will take care to ensure that the model is as easy as possible to design. 

## Functionality:
The functionality of this model at the current stage is to have a few modules. 
1. A Data Module that turns input data such as DNA sequence and Protein Concentration into numpy form which can be feed into the model for both training and testing. The DNA sequence can be string and the protein concentration can be numeric at the moment. We will consider further research and connection the future. 
2. A PWM module that construct the PWM part of the model. The PWM will take into account the difference in the footprint of the protein and the PWM. How this will be done will be up will be considered at a later date. 
3. A Barr-Algorithm module that implement the Barr Algorithm carefully. In this iterations, it will take into account cooperativity between the proteins.
4. A relationship module. It takes into the account the different roles of the TFs, either as activator, quencher, co-activators and co-quenchers in the form of a graph. As a result, the system will be capable of taking in a graph of this relationship and output a tf.model for optimization.
5. A combination module where 2. 3. and 4. are combined to construct a full model that can optimized. The optimization will be done using Tensorflow. 

There will be addition example codes and example database to study carefully. However, the author is not professional engineer and the design of the code can be further improved in the future when the needs arises.

## Examples: 
We are going to illustrate this using the following example. In the data folder there is a file `Data.npy` which is used in Liu, et. (2020). 
```python
#We first load the data into the right format
read_dictionary = np.load('Data.npy',allow_pickle = True).item()
Data = []
for i in read_dictionary.keys():
    DNA = read_dictionary['M3_2_lox']['Sequence']
    protein = {'eve':read_dictionary['M3_2_lox']['Eve_Concentration'],
          'dst':read_dictionary['M3_2_lox']['dst'],
          'cad':read_dictionary['M3_2_lox']['cad'], 
          'gt' :read_dictionary['M3_2_lox']['gt'],
          'dic':read_dictionary['M3_2_lox']['dic'],
          'hb' :read_dictionary['M3_2_lox']['hb'],
          'kni':read_dictionary['M3_2_lox']['kni'],
          'kr' :read_dictionary['M3_2_lox']['Kr'],
          'tll':read_dictionary['M3_2_lox']['tll'],
          'bcd':read_dictionary['M3_2_lox']['bcd']}
    Data += [DDC.Organism_data(DNA, protein, 'M3_2_lox')]

#Next we put in the protein information 
bcd_pwm = TF_PWM(name = 'bcd',PWM =np.array([[83,114,106,80],[74,159,72,78],[108,127,114,34],[48,149,11,175],[6,1,0,376],[381,0,2,0]\
                             ,[379,0,4,0],[5,0,4,374],[0,383,0,0],[6,340,3,34],[72,136,132,43],[61,174,60,88],[65,166,52,100],[68,158,49,108]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [0,0])
cad_pwm = TF_PWM(name = 'cad',PWM =np.array([[9,10,4,11],[12,6,4,16],[3,3,3,29],[4,0,0,34],[12,0,2,24],[38,0,0,0],[0,0,0,38],\
                            [4,0,7,27],[22,0,15,1],[1,8,10,1]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [2,2])
dst_pwm = TF_PWM(name = 'dst',PWM =np.array([[1,0,0,29],[1,0,1,28],[2,1,0,27],[1,27,1,1],[1,20,6,3],[5,16,8,1],[3,3,22,2],\
                            [0,2,27,1],[24,3,1,2],[28,0,1,1],[27,1,0,2],[5,8,6,11]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [1,1])
dic_pwm = TF_PWM(name = 'dic',PWM =np.array([[1,8,7,13],[0,25,0,4],[0,17,0,1],[20,0,0,9],[0,0,0,29],[0,0,3,26],[2,0,27,0],[0,0,0,29],\
                            [1,2,4,22],[4,10,6,9],[6,1,1,21]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [2,1])
Hb_pwm  = TF_PWM(name = 'hb',PWM = np.array([[53,6,224,7],[2,6,3,279],[0,2,0,288],\
                            [2,0,0,288],[0,2,0,288],[0,3,0,287],\
                            [0,2,0,288],[281,0,3,6],[31,43,78,138],\
                            [20,100,109,61]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [2,2])
gt_pwm  = TF_PWM(name = 'gt',PWM =np.array([[86,62,19,942],[12,108,359,630],[776,25,275,33],[8,762,65,274],\
                           [83,19,996,11],[0,556,0,553],[1020,88,1,0],[1106,0,0,3],[15,378,85,631]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [8,7])
Kr_pwm  = TF_PWM(name = 'kr',PWM = np.array([[17,73,6,101],[187,5,0,5],[158,39,0,0],[0,194,1,2],[1,194,0,2],\
                           [0,197,0,0],[8,22,6,161],[0,2,0,195],[2,34,2,159],[44,109,15,29]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [2,2])
kni_pwm = TF_PWM(name = 'kni',PWM =np.array([[19,1,2,4],[25,1,0,0],[16,0,0,10],[5,9,6,6],[0,4,1,21],[21,0,5,0],[0,0,26,0]\
                            ,[17,0,8,1],[1,3,18,4],[0,26,0,0],[25,0,1,0],[5,12,7,2]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [1,1])
tll_pwm = TF_PWM(name = 'tll',PWM = np.array([[12,8,0,0],[1,2,2,15],[1,2,1,16],[5,1,0,14],[2,3,15,0],[11,1,5,3],\
                            [1,17,0,2],[0,2,1,17],[0,3,2,15]]),\
                             background_frequency = np.array([0.297,0.203,0.203,0.297]), footprint_adjust = [3,2])

proteins = [bcd_pwm, cad_pwm, dst_pwm, dic_pwm, Hb_pwm, gt_pwm, Kr_pwm, kni_pwm,tll_pwm]

#Then we build the relationship as in Liu, et. (2020).

cooperativities = TF_TF_relationship(actors = ['bcd'], acted= 'bcd', name = 'bcd_bcd_coop',rtype = 'cooperativity', properties = {'range':60})
co_activations = TF_TF_relationship(actors = ['bcd','cad'],acted = 'hb', rtype ='coactivation',output_name = ['coactivated_hb','quenching_hb'],properties = {'range':150})
quenching_bcd = TF_TF_relationship(actors = ['gt','tll','kni','kr','quenching_hb' ], acted = 'bcd', rtype = 'quenching', output_name = 'quenching_bcd',properties = {'range':100})
quenching_cad = TF_TF_relationship(actors = ['gt','tll','kni','kr','quenching_hb' ], acted = 'cad', rtype = 'quenching', output_name = 'quenching_cad',properties = {'range':100})
quenching_dst = TF_TF_relationship(actors = ['gt','tll','kni','kr','quenching_hb' ], acted = 'dst', rtype = 'quenching', output_name = 'quenching_dst',properties = {'range':100})
quenching_dic = TF_TF_relationship(actors = ['gt','tll','kni','kr','quenching_hb' ], acted = 'dic', rtype = 'quenching', output_name = 'quenching_dic',properties = {'range':100})
quenching_coactivation_hb = TF_TF_relationship(actors = ['gt','tll','kni','kr','quenching_hb' ], acted = 'coactivated_hb', rtype = 'quenching', output_name = 'quenching_coactivated_hb',properties = {'range':100})
activation = tfr.TF_TF_relationship(actors = ['quenching_bcd','quenching_cad','quenching_dst','quenching_dic','quenching_coactivated_hb'], acted = 'eve', rtype = 'activation')
co_activations.next_relationships([quenching_bcd,quenching_cad,quenching_dst,quenching_dic,quenching_coactivation_hb ])
quenching_bcd.next_relationships([activation])
quenching_cad.next_relationships([activation])
quenching_dst.next_relationships([activation])
quenching_dic.next_relationships([activation])
quenching_coactivation_hb.next_relationships([activation])

relationship_list = TF_TF_relationship_list(relationships =[cooperativities,co_activations, quenching_bcd,\
                                                    quenching_cad, quenching_dst, quenching_dic,\
                                                   quenching_coactivation_hb,activation ], name_of_organism= 'Drosophila')

#Finally, we run the model. 
Trial_organism = Organism_models(proteins, relationship_list, 'eve')
Trial_organism.build_model()
X, Y = Trial_organism.build_fit_data(Data)
Trial_organism.model.compile(optimizer='rmsprop', loss='mse', metrics = 'mse')

#The model is compiled and can be fitted. (at the moment one data point at a time batch_size =1 )
Trial_organism.model.fit(
    x=X, y=Y, batch_size=10, epochs=10, verbose=1, callbacks=None,
    validation_split=0.2, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=2, validation_batch_size=None, validation_freq=1,
    max_queue_size=1, workers=1, use_multiprocessing=True
)


```





## References 
Liu Y, Barr K, Reinitz J. Fully interpretable deep learning model of transcriptional control[J]. Bioinformatics, 2020, 36(Supplement_1): i499-i507.

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.