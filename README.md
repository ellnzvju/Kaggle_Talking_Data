# Kaggle_Talking_Data
**11th** solutions for Kaggle - TalkingData Mobile user demographics

**Competition page (total prize $25000)**
[Competition Page](https://www.kaggle.com/c/talkingdata-mobile-user-demographics)

**Scoring: Multiclasses logloss**

The framework is based on 3 levels learning architecture, with stacking technique
-  1st level, consist of 4 models, the results are created from out of fold training and will be use for training in second level.
-  2nd level models, consist of XGBoost(gbtree) and NN(relu,relu,softmax) with 10 times bagging.
-  3rd level ensemble (weighted arimethic mean)

**All models in 1st layers are trained by the same fold random indices. (777)**

The final solution has been created by **running above architecture twice with 5 folds cross validation and 10 fold cross validation, then ensemble both results with geo-mean**

***Due to laptop limitation, I rarely run grid search. only small random grid search for Neural network parameters.**


#### Models and features used in 1st level training:

Features from features engineering process will be supplied into first level models (each models requires different,or similiar set of features). Train with 5 folds cv, 10 folds cv.
The probability result for each class from first level models will be supplied into second level learner.

**Features extraction** are primary divided features into three groups, before combine them into sparse train, test matrix for each model in first layer

1. *Device information* such device brand, device model. These features have been processed by one hot encoder.

2. *Events related information*. For instances, time of event, day of event, applications that are installed on target devices. However, from data characteristic, roughtly 69% of whole train and test don't have events related information. These features seem to have very good connection to user demographics, however, many data do not have these. All data have been processed by one hot encoder as well.

3. *Row ID*. From exploit found by other kaggler, the train/test data in this competition is not shuffle. So rowID is exploit hole in accuracy and can roughly improve log loss score by ~ 0.05 - 0.08 which is very critical. RowID have been extracted with many approaches like normalization, sorting, sorting both train/test, sorting with device_id as second sort index, clustering 10000, clustering 20000,and clustering 50000.

Without normalize(rowID) feature, mostly any single model, that are created by neural network (Keras), provide significantly better local score than gradient boost from XGB (gblinear). (0.03 different)

However, after with normalize(rowID), XGB models tend to get better local score than neural network. (0.01 - 0.02 different)

#### Models in 2nd level training:
All features come from 1st models prediction. (out of fold for training). In second level, XGB and NN are main learning approaches. With 10 rounds bagging, all seed are random in every rounds.

XGB result got around 2.16 on public LB (For both 5 folds and 10 folds settings)
NN result got around 2.18 , 2.17 on public LB (5, 10 folds respectively)

#### Ensemble:
Weight arimethic mean 70:30 for XGB:NN (improve result by little 0.003- 0.004 on public LB)


#### Final model:
created by geo mean between 5 folds 3rd level result and 10 folds 3rd level result, it leads to 2.15 in public LB and 2.16 in private LB

#### Final Thought:
Overall, it is pretty good competition, participant has to deal with many "lack of quality" data. However, with exploit on rowID on 3 days to the end, it leads to many problem. Many competitors has to rerun whole learning architectures again.
Some people who uses grid search will have problem. they wasted their time.

With better hardware, if I can perform fully grid search, expected around 0.01 - 0.03 increased in LB score.
