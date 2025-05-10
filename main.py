from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable, Any, Sized

import matplotlib.pyplot as plt
import optuna
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from optuna import Trial
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing_extensions import Optional, Literal

MODEL_KEY = 'model'
HISTORY_KEY = 'learning_history'
TRAIN_SCORE_KEY = 'train_score'
PARAMS_COUNT_KEY = 'params_count'
LAYERS_COUNT_KEY = 'layers_count'
LAYER_TYPE_KEY = 'layer_type'
EPSILON = 1e-12


def vec(*elems) -> NDArray:
    return np.array(elems)


@dataclass(init=False)
class Dataset:
    args: NDArray
    answers: NDArray

    def __init__(self, args: NDArray, answers: NDArray):
        assert args.ndim == 2
        assert answers.ndim == 1
        assert len(args) == len(answers)
        self.args = args
        self.answers = answers

    @property
    def features_count(self) -> int:
        return self.args.shape[1]

    def __len__(self) -> int:
        return len(self.args)


def linear[T](x: T) -> T:
    return x


def d_linear(x: NDArray) -> NDArray:
    return np.ones(x.shape)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - (tanh(x)) ** 2


def ReLU(x):
    return x * (x > 0)


def d_ReLU(x):
    return (x > 0) * np.ones(x.shape)


@dataclass
class Activation:
    __function__: Callable[[NDArray], NDArray]
    __derivative__: Callable[[NDArray], NDArray]

    def __call__(self, x: NDArray) -> NDArray:
        result = self.__function__(x)
        assert result.shape == x.shape
        return result

    def derivative(self, x: NDArray) -> NDArray:
        result = self.__derivative__(x)
        assert result.shape == x.shape
        return result


IDENTITY_ACTIVATION = Activation(linear, d_linear)
LINEAR_ACTIVATION = Activation(linear, d_linear)
RELU_ACTIVATION = Activation(ReLU, d_ReLU)
TANH_ACTIVATION = Activation(tanh, d_tanh)


def stable_softmax(z: NDArray) -> NDArray:
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def loss(predicted: NDArray, correct: NDArray) -> float:
    assert predicted.ndim == 2
    assert correct.ndim == 1
    assert len(predicted) == len(correct)

    one_hot = class_code_to_boolean_array(correct, predicted.shape[1])
    probs = stable_softmax(predicted)
    probs_clipped = np.clip(probs, EPSILON, 1. - EPSILON)
    return -np.sum(one_hot * np.log(probs_clipped)) / len(predicted)


def d_loss(predicted: NDArray, correct: NDArray) -> NDArray:
    assert predicted.ndim == 2
    assert correct.ndim == 1
    assert len(predicted) == len(correct)
    one_hot = class_code_to_boolean_array(correct, predicted.shape[1])
    probs = stable_softmax(predicted)
    return (probs - one_hot) / len(correct)


@dataclass
class CostFunction:
    __function__: Callable[[NDArray, NDArray], float]
    __derivative__: Callable[[NDArray, NDArray], NDArray]

    def __call__(self, predicted: NDArray, correct: NDArray) -> float:
        return self.__function__(predicted, correct)

    def derivative(self, predicted: NDArray, correct: NDArray) -> NDArray:
        return self.__derivative__(predicted, correct)


CROSS_ENTROPY_COST = CostFunction(loss, d_loss)


def class_code_to_boolean_array(arr: NDArray, classes_count: int) -> NDArray:
    assert arr.ndim == 1
    result = np.zeros((len(arr), classes_count))
    result[range(len(arr)), arr] = 1
    return result


class AdamOptimizer:
    momentum1: float
    momentum2: float
    epsilon: float
    vdW: NDArray
    vdb: NDArray
    SdW: NDArray
    Sdb: NDArray
    calls_count: int

    def __init__(self, shape_W: tuple[int, ...], shape_b: tuple[int, ...],
                 momentum1: float = 0.9, momentum2: float = 0.999, epsilon: float = 1e-8):
        assert len(shape_W) == 2
        assert len(shape_b) == 1
        assert shape_W[1] == shape_b[0]

        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.epsilon = epsilon

        self.vdW = np.zeros(shape_W)
        self.vdb = np.zeros(shape_b)

        self.SdW = np.zeros(shape_W)
        self.Sdb = np.zeros(shape_b)
        self.calls_count = 0

    # noinspection PyAugmentAssignment
    def get_next(self, dW: NDArray, db: NDArray) -> tuple[NDArray, NDArray]:
        self.vdW = self.momentum1 * self.vdW + (1 - self.momentum1) * dW
        self.vdb = self.momentum1 * self.vdb + (1 - self.momentum1) * db
        self.SdW = self.momentum2 * self.SdW + (1 - self.momentum2) * (dW ** 2)
        self.Sdb = self.momentum2 * self.Sdb + (1 - self.momentum2) * (db ** 2)
        vdW_h = self.vdW
        vdb_h = self.vdb
        SdW_h = self.SdW
        Sdb_h = self.Sdb
        if self.calls_count > 1:
            vdW_h = vdW_h / (1 - (self.momentum1 ** self.calls_count))
            vdb_h = vdb_h / (1 - (self.momentum1 ** self.calls_count))
            SdW_h = SdW_h / (1 - (self.momentum2 ** self.calls_count))
            Sdb_h = Sdb_h / (1 - (self.momentum2 ** self.calls_count))
        den_W = np.sqrt(SdW_h) + self.epsilon
        den_b = np.sqrt(Sdb_h) + self.epsilon

        self.calls_count += 1

        return vdW_h / den_W, vdb_h / den_b


type WeightsFactory = Callable[[int, int], NDArray]


class Layer(ABC):
    neurons: int

    @abstractmethod
    def initialize_parameters(self, input_dim: int):
        pass

    @abstractmethod
    def forward(self, X: NDArray) -> NDArray:
        pass

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        pass

    @abstractmethod
    def backpropagation(self, da: NDArray):
        pass

    @abstractmethod
    def update(self, lr: float):
        pass

    @property
    @abstractmethod
    def parameters_count(self) -> int:
        pass


import numpy as np
from typing import Callable
from numpy.typing import NDArray
from dataclasses import dataclass


class RBFLayer(Layer):
    def __init__(self, neurons: int, gamma: float, weights_factory: WeightsFactory):
        self.neurons = neurons
        self.gamma = gamma
        self.weights_factory = weights_factory

    def initialize_parameters(self, input_dim: int):
        self.W = self.weights_factory(input_dim, self.neurons)

    def forward(self, X: NDArray) -> NDArray:
        self.X = X
        X_norm2 = np.sum(X ** 2, axis=1, keepdims=True)
        W_norm2 = np.sum(self.W ** 2, axis=0, keepdims=True)
        d2 = X_norm2 + W_norm2 - 2 * (X @ self.W)
        self.A = np.exp(-self.gamma * d2)
        return self.A

    def predict(self, X: NDArray) -> NDArray:
        X_norm2 = np.sum(X ** 2, axis=1, keepdims=True)
        W_norm2 = np.sum(self.W ** 2, axis=0, keepdims=True)
        d2 = X_norm2 + W_norm2 - 2 * (X @ self.W)
        return np.exp(-self.gamma * d2)

    def backpropagation(self, dA: NDArray) -> NDArray:
        U = dA * self.A
        factor = 2 * self.gamma
        U2 = factor * U

        sum_U2_over_i = np.sum(U2, axis=0, keepdims=True)
        self.dW = self.X.T @ U2 - self.W * sum_U2_over_i

        sum_U2_over_j = np.sum(U2, axis=1, keepdims=True)
        dX = U2 @ self.W.T - sum_U2_over_j * self.X

        return dX

    def update(self, lr: float):
        self.W -= lr * self.dW

    @property
    def parameters_count(self) -> int:
        return self.W.size


class StandardLayer(Layer):
    W: NDArray
    X: NDArray
    activation: Activation
    b: NDArray
    dW: NDArray
    db: NDArray
    input_dim: int
    neurons: int
    optimizer: AdamOptimizer
    z: NDArray

    def initialize_parameters(self, input_dim: int):
        self.W = self.weights_factory(input_dim, self.neurons)
        self.b = np.zeros(self.neurons)
        self.optimizer = AdamOptimizer(self.W.shape, self.b.shape)

    def __init__(self, neurons: int, activation: Activation,
                 weights_factory: WeightsFactory):
        self.neurons = neurons
        self.activation = activation
        self.weights_factory: WeightsFactory = weights_factory

    def forward(self, X: NDArray) -> NDArray:
        self.X = X
        r = X @ self.W
        t = self.b.T
        self.z = r + t
        return self.activation(self.z)

    def predict(self, X: NDArray) -> NDArray:
        return self.activation(X @ self.W + self.b.T)

    def backpropagation(self, da: NDArray):
        dz = da * self.activation.derivative(self.z)
        self.db = np.sum(dz, axis=0)
        self.dW = self.X.T @ dz
        return dz @ self.W.T

    def update(self, lr: float):
        dW, db = self.optimizer.get_next(self.dW, self.db)
        self.W -= dW * lr
        self.b -= db * lr

    @property
    def parameters_count(self) -> int:
        return self.W.size + self.b.size


def same_len(*sequences: Sized) -> bool:
    if len(sequences) == 0:
        return True
    length: int = len(sequences[0])
    for sequence in sequences:
        if length != len(sequence):
            return False
    return True


@dataclass
class LearningHistory:
    test: Dataset
    train_loss: list[float] = field(default_factory=list, init=False)
    test_loss: list[float] = field(default_factory=list, init=False)
    train_score: list[float] = field(default_factory=list, init=False)
    test_score: list[float] = field(default_factory=list, init=False)
    epochs: list[int] = field(default_factory=list, init=False)

    @property
    def final_test_score(self):
        return self.test_score[-1]

    @property
    def final_train_score(self):
        return self.train_score[-1]

    def record(
            self,
            epoch: int,
            test_model: Callable[[Dataset], tuple[float, float]],
            train_score: float,
            train_loss: float
    ):
        test_loss, test_score = test_model(self.test)
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_score.append(train_score)
        self.test_loss.append(test_loss)
        self.test_score.append(test_score)

    def plot(self, layer_type: Literal['standard', 'rbf', ''], layers_count: int):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='row', figsize=(10, 5))
        fig.suptitle(f"{layer_type} model with {layers_count} layers")
        self.__loss_plot__(ax1)
        self.__score_plot__(ax2)
        fig.show()

    def __loss_plot__(self, ax: plt.Axes):
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.plot(self.epochs, self.train_loss, 'k', label='Train')
        ax.plot(self.epochs, self.test_loss, 'r', label='Test')
        ax.set_title('Model Loss')
        ax.legend()

    def __score_plot__(self, ax: plt.Axes):
        ax.set_ylabel('Score')
        ax.set_xlabel('Epoch')
        ax.plot(self.epochs, self.train_score, 'k', label='Train')
        ax.plot(self.epochs, self.test_score, 'r', label='Test')
        ax.set_title('Model Accuracy')
        ax.legend()


def score(answers: NDArray, predictions: NDArray) -> float:
    assert answers.ndim == 1
    assert predictions.ndim == 1
    assert len(answers) == len(predictions)
    return f1_score(answers, predictions, average='weighted')


class MLP:
    layers: list[Layer]

    def __init__(self, in_sz: int, layers: list[Layer], cost: CostFunction):
        assert len(layers) > 0
        self.architecture = None
        self.layers = layers
        self.cost = cost
        for i, layer in enumerate(self.layers):
            layer.initialize_parameters(in_sz)
            in_sz = layer.neurons

    def forward(self, x: NDArray) -> NDArray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x: NDArray) -> NDArray:
        for layer in self.layers:
            x = layer.predict(x)
        return x

    def backward(self, dZ: NDArray) -> NDArray:
        assert dZ.ndim == 2
        for layer in reversed(self.layers):
            dZ = layer.backpropagation(dZ)
        return dZ

    def fit(
            self,
            train: Dataset,
            test: Dataset,
            epochs: int,
            learning_rates: Iterable[float],
    ) -> LearningHistory:
        assert epochs > 0

        history = LearningHistory(test)
        learning_rates_iterator = iter(learning_rates)

        for epoch in range(epochs):
            predictions = self.forward(train.args)
            self.backward(self.cost.derivative(predictions, train.answers))
            self.update(next(learning_rates_iterator))

            history.record(
                epoch,
                self.test_model,
                score(train.answers, predictions.argmax(axis=1)),
                self.cost(predictions, train.answers)
            )
        return history

    def update(self, lr: float):
        for layer in self.layers:
            layer.update(lr)

    def test_for_score(self, dataset: Dataset) -> float:
        return score(dataset.answers, self.predict(dataset.args).argmax(axis=1))

    def test_model(self, dataset: Dataset) -> tuple[float, float]:
        probabilities = self.predict(dataset.args)
        cost = self.cost(probabilities, dataset.answers)
        return cost, score(dataset.answers, probabilities.argmax(axis=1))

    @property
    def parameters_count(self):
        return sum((layer.parameters_count for layer in self.layers))


def get_dataset(random_state: int) -> tuple[Dataset, Dataset, int]:
    iris = load_iris()
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3,
        random_state=random_state
    )
    assert len(x_test_raw) <= len(x_train_raw)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)
    return Dataset(x_train, y_train), Dataset(x_test, y_test), len(iris['target_names'])


def split(dataset: Dataset, ratio: float) -> tuple[Dataset, Dataset]:
    x_train, x_test, y_train, y_test = train_test_split(dataset.args, dataset.answers, train_size=ratio)
    return Dataset(x_train, y_train), Dataset(x_test, y_test)


def prelu_activation(alpha: float = 0.2):
    def PReLU(x):
        return np.where(x > 0, x, alpha * x)

    def d_PReLU(x):
        return np.where(x > 0, 1, alpha)

    return Activation(PReLU, d_PReLU)


def suggest_activation(trial: Trial, i: int) -> Activation:
    match trial.suggest_categorical(f'activation_function_{i}', ('identity', 'linear', 'relu', 'tanh')):
        case 'identity':
            return IDENTITY_ACTIVATION
        case 'linear':
            return LINEAR_ACTIVATION
        case 'relu':
            return RELU_ACTIVATION
        case 'prelu':
            return prelu_activation(trial.suggest_float('alpha', 0, 1))
        case 'tanh':
            return TANH_ACTIVATION
    raise ValueError('Unknown activation function')


def epoch_based_lr(function: Callable[[int], float]) -> Iterable[float]:
    t = 0
    while True:
        yield function(t)
        t += 1


def constant_lrs(lr: float) -> Iterable[float]:
    while True:
        yield lr


def step_decay(t: int, lr_0: float, F: float, D: float) -> float:
    return lr_0 * (F ** np.floor((1 + t) / D))


def exponential_decay(t: int, lr_0: float, k: float) -> float:
    return lr_0 * np.exp(-k * t)


def time_decay(t: int, lr_0: float, k: float) -> float:
    return lr_0 / (1 + (k * t))


def suggest_lr(trial: Trial) -> Iterable[float]:
    starting_lr = trial.suggest_float('starting_learning_rate', 1e-5, 1e-1, log=True)
    match trial.suggest_categorical('learning_rate_type',
                                    ['constant', 'time_decaying', 'step_decaying', 'exponential_decaying']):
        case 'constant':
            return constant_lrs(starting_lr)
        case 'time_decaying':
            k = trial.suggest_float('k', 1, 10)
            return epoch_based_lr(lambda t: time_decay(t, starting_lr, k))
        case 'step_decaying':
            F = trial.suggest_float('F', 0, 1)
            D = trial.suggest_float('D', 0.0001, 10)
            return epoch_based_lr(lambda t: step_decay(t, starting_lr, F, D))
        case 'exponential_decaying':
            k = trial.suggest_float('k', 1, 10)
            return epoch_based_lr(lambda t: exponential_decay(t, starting_lr, k))
    raise ValueError("unsupported lr type")


def show_trial(
        params: dict[str, Any],
        history: LearningHistory,
        layer_type: Literal['standard', 'rbf', '']
):
    print(f'Printing trial {layer_type}:')
    for name, value in params.items():
        print(f'    {name} = {value}')
    print('Constructing plot of learning history')
    layers_count = params['layers_count']
    assert isinstance(layers_count, int)
    history.plot(layer_type, layers_count)


def random_normal_initializer(n: int, m: int) -> NDArray:
    return np.random.standard_normal((n, m))


def he_initializer(n: int, m: int) -> NDArray:
    return np.random.uniform(0, 2 / n, (n, m))


def xavier_initializer(n: int, m: int) -> NDArray:
    limit = np.sqrt(6 / (n + m))
    return np.random.uniform(-limit, limit, (n, m))


def suggest_initializer(trial: Trial, i: int) -> WeightsFactory:
    match trial.suggest_categorical(f'weights_initializer_{i}', ('he', 'xavier')):
        case 'random':
            return random_normal_initializer
        case 'he':
            return he_initializer
        case 'xavier':
            return xavier_initializer
    raise ValueError('Unknown initializer type')


def suggest_layer(
        trial: Trial,
        i: int,
        max_params: int,
        layer_type: Literal['standard', 'rbf', ''] = '',
        neurons_count: int = 0
) -> Layer:
    assert max_params > 0
    if neurons_count <= 0:
        neurons_count = trial.suggest_int(f'neurons_count_{i}', 1, max_params)
    weights_factory = suggest_initializer(trial, i)
    trial.set_user_attr(LAYER_TYPE_KEY, layer_type)
    if len(layer_type) == 0:
        layer_type = trial.suggest_categorical(f'layer_type_{i}', ['standard', 'rbf'])
    match layer_type:
        case 'standard':
            return StandardLayer(
                neurons_count,
                suggest_activation(trial, i),
                weights_factory
            )
        case 'rbf':
            return RBFLayer(
                neurons_count,
                trial.suggest_float('rbf_gamma', 1e-5, 1e1, log=True),
                weights_factory
            )
    raise ValueError('Unknown layer type')


def suggest_model(
        trial: Trial,
        features_count: int,
        classes_count: int,
        max_params: int,
        max_layers: int,
        layer_type: Literal['standard', 'rbf', ''] = ''
) -> MLP:
    layers_count = trial.suggest_int('layers_count', 1, max_layers)

    layers = [suggest_layer(trial, i, max_params, layer_type) for i in range(layers_count - 1)]
    layers.append(suggest_layer(trial, layers_count - 1, max_params, layer_type, classes_count))

    return MLP(features_count, layers, CROSS_ENTROPY_COST)


class ParametersCountException(Exception):
    maximum: int
    actual: int

    def __init__(self, maximum: int, actual: int):
        self.maximum = maximum
        self.actual = actual

    def __str__(self):
        return f"Maximal count of parameters is {self.maximum}, but actual count is {self.actual}"


class TooHighScoreException(Exception):
    actual_score: float

    def __init__(self, actual_score: float):
        self.actual_score = actual_score

    def __str__(self):
        return f"Score is too high: {self.actual_score}"


def choose_trial(best_trials: list[FrozenTrial]) -> FrozenTrial:
    assert len(best_trials) > 0
    result: Optional[FrozenTrial] = None

    def is_better(result: FrozenTrial, best_trial: FrozenTrial) -> bool:
        if result is None:
            return True

        if result.values[0] >= 1.0 and best_trial.values[1] < result.values[1]:
            return True

        if best_trial.values[0] >= 1.0:
            return best_trial.values[1] < result.values[1]

        if best_trial.values[0] == result.values[0] and best_trial.values[1] < result.values[1]:
            return True

        return best_trial.values[0] > result.values[0]

    for best_trial in best_trials:
        if is_better(result, best_trial):
            result = best_trial

    assert result is not None
    return result


def find_best_model(
        max_params: int,
        max_layers: int,
        layer_type: Literal['standard', 'rbf', ''] = ''
) -> FrozenTrial:
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    train, test, classes_count = get_dataset(RANDOM_STATE)
    study = optuna.create_study(
        study_name='study optimal params',
        directions=(StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE)
    )

    def objective(trial: Trial) -> tuple[float, int, float]:
        suggested_model = suggest_model(trial, train.features_count, classes_count, max_params, max_layers, layer_type)
        learning_rates = suggest_lr(trial)
        epochs = trial.suggest_int('epochs', 1, 200)

        history = suggested_model.fit(train, test, epochs, learning_rates)

        trial.set_user_attr(HISTORY_KEY, history)
        trial.set_user_attr(PARAMS_COUNT_KEY, suggested_model.parameters_count)
        trial.set_user_attr(LAYERS_COUNT_KEY, len(suggested_model.layers))

        return history.final_test_score, suggested_model.parameters_count, history.final_train_score

    study.optimize(objective, timeout=60 * 2, n_jobs=-1, catch=(ParametersCountException, TooHighScoreException))
    return choose_trial(study.best_trials)


def extract_history(trial: FrozenTrial) -> LearningHistory:
    result = trial.user_attrs[HISTORY_KEY]
    assert isinstance(result, LearningHistory)
    return result


def extract_params_count(trial: FrozenTrial) -> int:
    result = trial.user_attrs[PARAMS_COUNT_KEY]
    assert isinstance(result, int)
    return result


def extract_layers_count(trial: FrozenTrial) -> int:
    result = trial.user_attrs[LAYERS_COUNT_KEY]
    assert isinstance(result, int)
    return result


HISTORY_NUMPY_STRUCT = np.dtype([('layers', int), ('params', int), ('train_score', float), ('test_score', float)])


@dataclass(init=False)
class LayersHistory:
    history: NDArray

    def __init__(self, size: int):
        self.history = np.zeros(size, dtype=HISTORY_NUMPY_STRUCT)

    def __len__(self) -> int:
        return len(self.history)

    def plot(self, layer_type: str, axs: NDArray[Axes], i: int):
        layers_ax = axs[i, 0]
        assert isinstance(layers_ax, Axes)
        self.plot_layers(layers_ax, layer_type)

        params_ax = axs[i, 1]
        assert isinstance(params_ax, Axes)
        self.plot_params(params_ax, layer_type)

    def plot_layers(self, ax: Axes, layer_type: str):
        self.history.sort(order='layers')

        ax.set_ylabel('Score')
        ax.set_xlabel('Layers count')

        layers_count = self.history['layers']
        ax.plot(layers_count, self.history['train_score'], 'k', label='Train')
        ax.plot(layers_count, self.history['test_score'], 'r', label='Test')

        ax.set_title(
            'Layers count vs Model score' if len(layer_type) == 0 else f'{layer_type} layers count vs Model score'
        )
        ax.legend()

    def plot_params(self, ax: Axes, layer_type: str):
        self.history.sort(order='params')

        ax.set_ylabel('Score')
        ax.set_xlabel('Parameters count')

        params_count = self.history['params']
        ax.plot(params_count, self.history['train_score'], 'k', label='Train')
        ax.plot(params_count, self.history['test_score'], 'r', label='Test')

        ax.set_title(
            'Parameters count vs Model score' if len(
                layer_type) == 0 else f'{layer_type} parameters count vs Model score'
        )
        ax.legend()

    def record(self, i: int, trial: FrozenTrial):
        learning_history = extract_history(trial)
        self.history[i] = (
            extract_layers_count(trial),
            extract_params_count(trial),
            learning_history.final_train_score,
            learning_history.final_test_score
        )

    def dumb_data(self):
        rng = np.random.default_rng()

        self.history['layers'] = rng.uniform(low=1, high=20, size=len(self))
        self.history['params'] = rng.uniform(low=1, high=200, size=len(self))
        self.history['train_score'] = rng.random(len(self))
        self.history['test_score'] = rng.random(len(self))


def layers_history(
        data: NDArray,
        layer_type: Literal['standard', 'rbf', '']
) -> LayersHistory:
    history = LayersHistory(len(data))
    for i, (layers_limit, max_neurons) in enumerate(data):
        best_trial = find_best_model(max_neurons, layers_limit, layer_type)
        show_trial(best_trial.params, extract_history(best_trial), layer_type)
        print(f"Test score = {best_trial.values[0]}; Parameters count = {extract_params_count(best_trial)}")
        history.record(i, best_trial)
    # history.dumb_data()
    return history


def print_best_model(layer_type: Literal['standard', 'rbf', ''] = ''):
    best_trial = find_best_model(60, 15, layer_type)
    show_trial(best_trial.params, extract_history(best_trial), layer_type)
    print(f"Test score = {best_trial.values[0]}; Parameters count = {extract_params_count(best_trial)}")


def my_subplots(rows: int, cols: int) -> tuple[Figure, NDArray[Axes]]:
    return plt.subplots(3, 2, figsize=(cols * 5, rows * 5))


def layers_count_analysis():
    layer_types: list[Literal['standard', 'rbf', '']] = ['standard', 'rbf', '']
    fig, axs = my_subplots(len(layer_types), 2)

    for i, layer_type in enumerate(layer_types):
        layers_history(vec([1, 240], [4, 60], [10, 60], [20, 60]), layer_type).plot(layer_type, axs, i)
    fig.show()


def main():
    print_best_model('standard')
    print_best_model('rbf')
    print_best_model()
    print("============== Studying parameters count ==============")
    layers_count_analysis()


if __name__ == '__main__':
    main()

'''
Good hyperparameters
Printing trial:
    layers_count = 3
    neurons_count_0 = 46
    weights_initializer_0 = xavier
    layer_type_0 = rbf
    rbf_gamma = 0.9437271912678421
    neurons_count_1 = 48
    weights_initializer_1 = xavier
    layer_type_1 = standard
    activation_function_1 = identity
    weights_initializer_2 = xavier
    layer_type_2 = standard
    activation_function_2 = linear
    epochs = 112
    starting_learning_rate = 0.09442275248887276
    learning_rate_type = constant
Constructing plot of learning history
Best score:  0.9351025968497232
'''
