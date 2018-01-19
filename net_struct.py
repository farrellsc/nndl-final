from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import rmsprop

class FCN:
    def __init__(self, input_shape: list, output_shape: list, learning_rate=0.001):
        """
        input_shape: [3 * 8 * 8]
        output_shape: [64]
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.model = None
        self.build_network()

    def build_network(self) -> None:
        X = Input(self.input_shape, name='input1')
        flow = Dense(256, activation='sigmoid', input_shape=self.input_shape, name='dense1')(X)
        flow = Dense(128, activation='sigmoid', name='dense2')(flow)
        action_probs = Dense(64, name='dense3')(flow)
        model = Model(input=X, output=action_probs)
        optimizer = rmsprop(lr=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        self.model = model

class ConvNet:
    def __init__(self, input_shape: list, output_shape: list, learning_rate=0.001) -> None:
        """
        input_shape: [3, 8, 8]
        output_shape: [64]
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.model = None
        self.build_network()

    def build_network(self) -> None:
        X = Input(self.input_shape, name='input1')
        flow = Dense(256, activation='relu', input_shape=self.input_shape, name='dense1')(X)
        flow = Dense(128, activation='relu', name='dense2')(flow)
        action_probs = Dense(64, activation='softmax', name='dense3')(flow)
        model = Model(input=X, output=action_probs)
        optimizer = rmsprop(lr=self.learning_rate, decay=1e-6, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        self.model = model