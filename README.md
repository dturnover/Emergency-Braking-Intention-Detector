An emergency braking intention detector for motorists.

I trained a support vector machine on a dataset which recorded the brain signals of participants in a driving simulation. Participants wore EEG headsets and drove for 20 miutes in a driving simulation. The helmets recorded data streams consisting of 64 electroencephalogram signals. I used supervised learning and a sliding window to segment the streams into windows which were ascribed a label of 1 for about to brake and 0 for not about to brake. I achieved 90% true positive rate and a 92% true negative rate after testing the model with the test set. 

Must be run using matlab

An EEG dataset consisting of 64 signals must be fed to the program by changing the directory on line 1 to the directory where your data is stored
