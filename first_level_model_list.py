from neural_network_model_catalog import neural_network_catalog
import xgboost as xgb
from Wrapper.model_wrapper import KerasClassifier, XgboostClassifier

"""
All model will be wrapped by wrapper class (for keras and xgb)

Model support on Keras and XGB only... will add more in future.

- Xgbboost on all features
- NN


"""
"""
#[
All tested model
return [    ('mixlayers_prelu_tanh_without_norm', neural_network_catalog.mix_layers_without_normalization()),
            ('base_line_normal',neural_network_catalog.baseline_model_customable()),
            ('prelu_extension',neural_network_catalog.Prelu_extension()),
            ('prelu_high_dp',neural_network_catalog.prelu_network_2layers(hn1=128, hn2=128, dp=0.5)),
            ('leaky_2layers_units_alpha_adjusted', neural_network_catalog.leaky_2layers_units(hn1 = 100, hn2 = 150, dp = 0.4, alpha = 0.25)),
            ('relu_ultra_high_unit', neural_network_catalog.prelu_network_2layers(hn1=500, hn2= 500, dp=0.5)),
            ('relu_high_unit', neural_network_catalog.prelu_network_2layers(hn1=150, hn2= 64, dp=0.423)),
            ('leaky_2layers_high_units', neural_network_catalog.leaky_2layers_units(hn1 = 128, hn2 = 128, dp = 0.5, alpha = 0.23))]

"""


class model_list:

    """ Create model list to test each model sequencely """
    """ Observation from experiment
        - high number of hidden units (with PReLU), model tend to be overfit to train data size. but lost a lot in validation
          using few epoch may help to avoid overfit. but could lead to underfit. need to balance out.
        - High dropout helps overfit.
        - from varieties of models, not much win different. some model converge very fast, while some take more time.
        - for train data set, high hidden unit helps converging with high dropout would be great.

        Public leader board has close data distribution to train set. So it could be highly possibility to change when competition end.
    """
    nb_output = 12
    #nb_input = 21396
    nb_input = 21551
    #nb_input = 21554
    #nb_input = 2
    #nb_input = 2160
    NEURAL = 1
    XGBOOST = 2
    # model type
    # 0 - Keras, need build function
    # 1 - Xgboost, build function will be None
    # 2 - Other
    models = {
                'nn_base_line_normal': (neural_network_catalog.baseline_model_customable, {
                'hn1' : 24,
                'hn2' : 36,
                'dp'  : 0.2,
                'nb_epoch' : 15,
                'batch_size': 32,
                }, NEURAL),

                'nn_leaky_small_normal': (neural_network_catalog.baseline_model_customable, {
                'hn1' : 4,
                'hn2' : 4,
                'dp'  : 0.02,
                'nb_epoch' : 100,
                'batch_size': 500,
                }, NEURAL),

                'nn_base_line_increased': (neural_network_catalog.baseline_model_customable, {
                'hn1' : 180,
                'hn2' : 60,
                'dp'  : 0.23,
                'nb_epoch' : 14,
                'batch_size': 70,
                }, NEURAL),

                'nn_one_layer': (neural_network_catalog.one_layer, {
                'nb_epoch' : 50,
                'batch_size': 100,
                }, NEURAL),

                'nn_relu': (neural_network_catalog.relu_network_2layers, {
                'hn1' : 150,
                'hn2' : 50,
                'dp'  : 0.32, #0.3 get around 2.2434 , 0.2 is to low
                'nb_epoch' : 8, # without row
                #'nb_epoch' : 7, # for Row
                'batch_size': 250,
                }, NEURAL),

                'nn_relu_brand': (neural_network_catalog.relu_network_2layers, {
                'hn1' : 150,
                'hn2' : 50,
                'dp'  : 0.1,
                'nb_epoch' : 15,
                'batch_size': 50,
                }, NEURAL),

                'nn_prelu_extension': (neural_network_catalog.prelu_extension, {
                'nb_epoch' : 100,
                'batch_size': 500,
                }, NEURAL),

                'nn_base_line_default': (neural_network_catalog.baseline_model_customable, {
                'nb_epoch' : 15,
                'batch_size': 32,
                }, NEURAL),

                'nn_prelu_high_dp': (neural_network_catalog.prelu_network_2layers, {
                'hn1' : 500,
                'hn2' : 500,
                'dp'  : 0.5,
                'nb_epoch' : 10,
                'batch_size' : 400,
                }, NEURAL),

                'nn_leaky_2layers_units_alpha_adjusted': (neural_network_catalog.leaky_2layers_units,{
                'hn1': 150,
                'hn2': 50,
                'dp': 0.3,
                'alpha': 0.25,
                'nb_epoch': 10,
                'batch_size': 250,
                }, NEURAL),
                #good
                'three_layers': (neural_network_catalog.three_layers,{
                'nb_epoch': 7, #with adadelta reach 2.2559
                #'nb_epoch': 20,
                'batch_size': 500,
                #'batch_size': 250,
                }, NEURAL),

                'xgb_high_col': (None,{
                'objective':'multi:softprob',
                'eta': 0.05,
                'max_depth': 9,
                'eval_metric': 'mlogloss',
                'num_round': 950,
                'silent' : 1,
                'nthread': 4,
                'early_stopping_rounds': 100
                }, XGBOOST),

                'xgb_gblinear_extra_deep': (None,{
                'objective':'multi:softprob',
                'booster': 'gblinear',
                'eta': 0.01,
                'max_depth': 12,
                'eval_metric': 'mlogloss',
                'num_round': 2500,
                'colsample_bytree': 1.0,
                'alpha': 3,
                'silent' : 1,
                'nthread': 4,
                'early_stopping_rounds': 50
                }, XGBOOST),

                'xgb_dart_extra_deep': (None,{
                'objective':'multi:softprob',
                'booster': 'dart',
                'eta': 0.05,
                'max_depth': 22,
                'eval_metric': 'mlogloss',
                'num_round': 1000,
                'silent' : 1,
                'early_stopping_rounds': 50
                }, XGBOOST),

                }

    @staticmethod
    def _create_model(unique_id):
        if unique_id in model_list.models:
            if model_list.models[unique_id][2] == 1:
                return KerasClassifier(unique_id, model_list.models[unique_id][0],
                                        model_list.nb_input,
                                        model_list.nb_output,
                                        **model_list.models[unique_id][1])
            elif model_list.models[unique_id][2] == 2:
                return XgboostClassifier(unique_id, model_list.models[unique_id][0],
                                        model_list.nb_input,
                                        model_list.nb_output,
                                        **model_list.models[unique_id][1])
        else:
            raise Exception('Model name not found')


    @staticmethod
    def ModelLookUp(unique_id):
        return model_list._create_model(unique_id)
