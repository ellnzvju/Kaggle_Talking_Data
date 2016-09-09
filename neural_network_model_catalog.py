from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ThresholdedReLU, LeakyReLU
from keras.optimizers import Adadelta, Adam, rmsprop, SGD
from keras.utils import np_utils


class neural_network_catalog():

    # score 2.2459 in public board (with bagging ensemble and default setting)
    # @staticmethod
    # def baseline_model_customable(nb_input, nb_output, hn1 = 16, hn2 = 64, dp = 0.2, **kwargs):
    #     """ Provide customable on number of hidden units in layer 1/2
    #         As well as drop out size in connection between layer 1 and 2
    #     """
    #     # create model
    #     model = Sequential()
    #     model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform', activation='relu'))
    #     model.add(Dropout(dp))
    #     model.add(Dense(hn2, init='glorot_uniform', activation='tanh'))
    #     model.add(Dropout(dp))
    #     model.add(Dense(nb_output, init='glorot_uniform', activation='softmax'))
    #     # Compile model
    #     sgd = SGD(lr=0.03)
    #     model.compile(loss='categorical_crossentropy', optimizer=sgd)  #logloss
    #     return model

    @staticmethod
    def baseline_model_customable(nb_input, nb_output, hn1 = 16, hn2 = 64, dp = 0.2, **kwargs):
        """ Provide customable on number of hidden units in layer 1/2
            As well as drop out size in connection between layer 1 and 2
        """
        # create model
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform', activation='relu'))
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='glorot_uniform', activation='tanh'))
        model.add(Dropout(0.1))
        model.add(Dense(nb_output, init='glorot_uniform', activation='softmax'))
        # Compile model
        sgd = SGD(lr=0.03, decay =0.0001)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')  #logloss
        return model

    @staticmethod
    def one_layer(nb_input, nb_output, **kwargs):
        # create model
        model = Sequential()
        model.add(Dense(6, input_dim= nb_input, init='normal', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(nb_output, init='normal', activation='softmax'))
        # Compile model
        sgd = SGD(lr=0.08, decay =0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)  #logloss
        return model

    @staticmethod
    def three_layers(nb_input, nb_output, **kwargs):
        """ Provide customable on number of hidden units in layer 1/2
            As well as drop out size in connection between layer 1 and 2
        """
        # create model
        model = Sequential()
        model.add(Dense(210, input_dim= nb_input, init='glorot_uniform', activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(100, init='normal', activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(50, init='glorot_uniform', activation='linear'))
        model.add(Dense(nb_output, init='normal', activation='softmax'))
        # Compile model
        sgd = SGD(lr=0.08, decay =0.0001)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')  #logloss
        return model

    @staticmethod
    def baseline_model_sgd_optimization(nb_input, nb_output,hn1 = 16, hn2 = 64, dp = 0.2, lr = 0.001, mn = 0.1, **kwargs):
        """ use sgd with small learning rate to optimize network instead of adadelta
        """
        # create model
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='normal', activation='relu'))
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='normal', activation='tanh'))
        model.add(Dense(nb_output, init='normal', activation='softmax'))
        # Compile model
        sgd = SGD(lr=lr,  momentum=mn)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  #logloss
        return model

    @staticmethod
    def prelu_network_2layers(nb_input, nb_output, hn1 = 16, hn2 = 64, dp = 0.2, **kwargs):
        # create model
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(dp))
        model.add(Dense(nb_output, init='glorot_uniform', activation='softmax'))
        # Compile model
        sgd = SGD(lr=0.03, decay =0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)  #logloss
        return model

    @staticmethod
    def relu_network_2layers(nb_input, nb_output, hn1 = 16, hn2 = 64, dp=0.2, **kwargs):
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform', activation='relu'))
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='glorot_uniform', activation='relu'))
        model.add(Dropout(dp))
        model.add(Dense(nb_output, init='glorot_uniform', activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')  #logloss
        return model

    @staticmethod
    def leaky_2layers_units(nb_input, nb_output, hn1 = 16, hn2 = 64, dp = 0.2, alpha = 0.3, **kwargs):
        # create model
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='normal'))
        model.add(LeakyReLU(alpha = alpha))
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='normal'))
        model.add(LeakyReLU(alpha = alpha))
        model.add(Dense(nb_output, init='normal', activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
        return model

    @staticmethod
    def prelu_extension(nb_input, nb_output, **kwargs):
        model = Sequential()
        model.add(Dense(1000, input_dim=nb_input, init='normal', activation='tanh'))
        model.add(Dropout(0.4))
        model.add(Dense(500, init='normal', activation='tanh'))
        model.add(Dense(12, init='normal', activation='relu'))
        model.add(Dense(12, init='normal', activation='softmax'))
        sgd = SGD(lr=0.04, decay =0.0001)
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')  #logloss
        return model

    @staticmethod
    def mix_layers_without_normalization(nb_input, nb_output, **kwargs):
        model = Sequential()
        model.add(Dense(200, input_dim=nb_input, init='normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(50, input_dim=nb_input, init='normal', activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(12, init='normal', activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
        return model

    @staticmethod
    def mix_layers_with_normalization(nb_input, nb_output, **kwargs):
        model = Sequential()
        model.add(Dense(200, input_dim=nb_input, init='normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(50, input_dim=nb_input, init='normal', activation='tanh'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(12, init='normal', activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
        return model
