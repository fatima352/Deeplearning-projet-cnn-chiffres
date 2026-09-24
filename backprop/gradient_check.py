import numpy as np
from model.layers import ReLU, Linear, Conv2D, MaxPool2D, Sigmoid
from model.cnn import CNN


def numerical_gradient(f, x, epsilon=1e-5):
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + epsilon
        plus = f()
        x[idx] = original - epsilon
        minus = f()
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * epsilon)
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def test_relu():
    x = np.random.randn(6, 6)
    layer = ReLU()
    dout = np.random.randn(6, 6)
    layer.forward(x)
    dx_analytic = layer.backward(dout)

    def loss_fn(): return np.sum(layer.forward(x) * dout)

    dx_numeric = numerical_gradient(loss_fn, x)
    print("ReLU     - erreur relative dx :", relative_error(dx_analytic, dx_numeric))


def test_linear():
    x = np.random.randn(20)
    layer = Linear(n_inputs=20, n_outputs=5)
    dout = np.random.randn(5)
    layer.forward(x)
    dx_analytic, dw_analytic, db_analytic = layer.backward(dout)

    def loss_fn(): return np.sum(layer.forward(x) * dout)

    dx_numeric = numerical_gradient(loss_fn, x)
    dw_numeric = numerical_gradient(loss_fn, layer.weights)
    db_numeric = numerical_gradient(loss_fn, layer.bias)
    print("Linear   - erreur relative dx :", relative_error(dx_analytic, dx_numeric))
    print("Linear   - erreur relative dw :", relative_error(dw_analytic, dw_numeric))
    print("Linear   - erreur relative db :", relative_error(db_analytic, db_numeric))


def test_maxpool():
    x = np.random.randn(1, 4, 4)
    layer = MaxPool2D(pool_size=2)
    dout = np.random.randn(1, 2, 2)
    layer.forward(x)
    dx_analytic = layer.backward(dout)

    def loss_fn(): return np.sum(layer.forward(x) * dout)

    dx_numeric = numerical_gradient(loss_fn, x)
    print("MaxPool2D- erreur relative dx :", relative_error(dx_analytic, dx_numeric))


def test_conv2d():
    x = np.random.randn(6, 6)
    layer = Conv2D(n_filters=2, kernel_size=3)
    out = layer.forward(x)
    dout = np.random.randn(*out.shape)
    dkernels_analytic = layer.backward(dout)

    def loss_fn(): return np.sum(layer.forward(x) * dout)

    dkernels_numeric = numerical_gradient(loss_fn, layer.kernels)
    print("Conv2D   - erreur relative dkernels :", relative_error(dkernels_analytic, dkernels_numeric))


def test_sigmoid():
    x = np.random.randn(10)
    layer = Sigmoid()
    dout = np.random.randn(10)
    layer.forward(x)
    dx_analytic = layer.backward(dout)

    def loss_fn(): return np.sum(layer.forward(x) * dout)

    dx_numeric = numerical_gradient(loss_fn, x)
    print("Sigmoid  - erreur relative dx :", relative_error(dx_analytic, dx_numeric))


def test_cnn_integration():
    np.random.seed(0)
    x = np.random.rand(28, 28)
    y = np.zeros(10)
    y[3] = 1
    model = CNN()

    def compute_loss():
        prediction = model.forward(x)
        return -np.sum(y * np.log(prediction + 1e-9))

    compute_loss()
    prediction = model.forward(x)
    dloss_dprob = prediction - y
    dkernels_analytic, dw_fc_analytic, db_fc_analytic = model.backward(dloss_dprob)

    dkernels_numeric = numerical_gradient(compute_loss, model.conv.kernels)
    dw_fc_numeric = numerical_gradient(compute_loss, model.fc.weights)
    db_fc_numeric = numerical_gradient(compute_loss, model.fc.bias)

    print("\n=== CNN complet (intégration) ===")
    print("erreur relative dkernels :", relative_error(dkernels_analytic, dkernels_numeric))
    print("erreur relative dw_fc    :", relative_error(dw_fc_analytic, dw_fc_numeric))
    print("erreur relative db_fc    :", relative_error(db_fc_analytic, db_fc_numeric))


if __name__ == "__main__":
    print("=== Tests couche par couche ===")
    test_relu()
    test_linear()
    test_maxpool()
    test_conv2d()
    test_sigmoid()
    test_cnn_integration()