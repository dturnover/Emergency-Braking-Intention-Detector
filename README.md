An emergency braking intention detector for motorists.

I trained a support vector machine on a dataset which recorded the brain signals of participants in a driving simulation. Participants wore EEG headsets and drove for 20 miutes in a driving simulation. The helmets recorded data streams consisting of 64 electroencephalogram signals. I used supervised learning and a sliding window to segment the streams into windows which were ascribed a label of 1 for 'about to brake' and 0 for 'not about to brake'. I also extracted features for the columns (individual signal) in each sliding glass window. Features included mean, standard deviation, activity, mobility, and complexity. I achieved a 90% true positive rate and a 92% true negative rate after evaluating the model with the test set. 

Must be run using matlab

An EEG dataset consisting of 64 signals must be fed to the program by changing the directory on line 1 to the directory where your data is stored

Link to dataset: https://mega.nz/file/g19BAbiS#8puF48I_66n2o5g1K86evw3QNNjK3SL3EkcoFjSChRs

Link to dataset description: https://lampx.tugraz.at/~bci/database/002-2016/description.txt

Unzip the files into a folder called 'Emergency_Braking_EEG_Data'. Make sure the matlab file is in the same directory as this folder before running.
