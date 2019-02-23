1. Reweighs each group feature with weight $\frac{1.}{\sqrt(|group|)}$ and thread each group feature as categorical
2. Convert integer features in 3' idiots way and add then as categorical:  
    $ floor(\log_{2}^{2}(I)) $  
    $ floor(\log_{2}(I)) $  
    $ floor(I) $
3. Also add counters as integer features
3. Use 5 vw models  
4. Average over all models with weights

As a result those files are created:
* model_[0-9].vw - trained models for making predictions
* ./train_full.vw ./train.vw ./validation.vw ./test.vw - datasets for final training, calibration and testing
* ./tmp/train/[0-9]* ./tmp/test/[0-9]* - chunks of original train and test datasets
* ./tmp/train/[0-9]*.vw ./tmp/test/[0-9]*.vw - chunks of train and test datasets with extracted features
* probas_[0-9].txt - probas of click for examples from test dataset for each 
* submission.txt - submission for kaggle 

Dependencies:
* numpy
* natsort

Launching pipeline:
1. preprocess.py
2. train.py
3. make_submission.py
