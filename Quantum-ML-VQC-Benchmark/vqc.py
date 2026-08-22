import numpy as np

N_QUBITS = 4
DIM = 2 ** N_QUBITS


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128)


def rot(phi, theta, omega):
    # Rot(phi,theta,omega) = RZ(omega) RY(theta) RZ(phi)  (PennyLane convention)
    return rz(omega) @ ry(theta) @ rz(phi)


def apply_single_qubit_gate_batch(state, gate, qubit, n_qubits=N_QUBITS):
    """state: (batch, 2**n_qubits) complex. gate: (2,2). qubit index 0..n_qubits-1."""
    batch = state.shape[0]
    shape = (batch,) + (2,) * n_qubits
    st = state.reshape(shape)
    st = np.moveaxis(st, qubit + 1, 1)          # bring target qubit to axis 1
    flat = st.reshape(batch, 2, -1)
    flat = np.einsum('ij,bjk->bik', gate, flat)
    st = flat.reshape(st.shape)
    st = np.moveaxis(st, 1, qubit + 1)
    return st.reshape(batch, -1)


def apply_cnot_batch(state, control, target, n_qubits=N_QUBITS):
    batch = state.shape[0]
    shape = (batch,) + (2,) * n_qubits
    st = state.reshape(shape)
    st = np.moveaxis(st, [control + 1, target + 1], [1, 2])
    out = st.copy()
    idx1 = [slice(None)] * st.ndim
    idx1[1] = 1
    sub = st[tuple(idx1)]                       # control==1 branch, shape (batch,2,...) target now axis1
    sub_flipped = np.flip(sub, axis=1)           # flip target bit
    out[tuple(idx1)] = sub_flipped
    out = np.moveaxis(out, [1, 2], [control + 1, target + 1])
    return out.reshape(batch, -1)


def angle_embedding(x_batch, n_qubits=N_QUBITS):
    """x_batch: (batch, n_qubits) features already scaled to radians. Returns statevector (batch, DIM)."""
    batch = x_batch.shape[0]
    state = np.zeros((batch, DIM), dtype=np.complex128)
    state[:, 0] = 1.0
    for q in range(n_qubits):
        g_angles = x_batch[:, q]
        # apply per-sample RY -> can't broadcast a single 2x2 gate across batch since angle differs per sample
        c = np.cos(g_angles / 2)
        s = np.sin(g_angles / 2)
        shape = (batch,) + (2,) * n_qubits
        st = state.reshape(shape)
        st = np.moveaxis(st, q + 1, 1)
        flat = st.reshape(batch, 2, -1)
        new0 = c[:, None] * flat[:, 0, :] - s[:, None] * flat[:, 1, :]
        new1 = s[:, None] * flat[:, 0, :] + c[:, None] * flat[:, 1, :]
        flat = np.stack([new0, new1], axis=1)
        st = flat.reshape(st.shape)
        st = np.moveaxis(st, 1, q + 1)
        state = st.reshape(batch, -1)
    return state


def strongly_entangling_layer_batch(state, params_layer, n_qubits=N_QUBITS):
    """params_layer: (n_qubits, 3) — same rotation for every sample in the batch (this is the trainable weight)."""
    for q in range(n_qubits):
        phi, theta, omega = params_layer[q]
        state = apply_single_qubit_gate_batch(state, rot(phi, theta, omega), q, n_qubits)
    # ring of CNOTs (range = 1)
    for q in range(n_qubits):
        state = apply_cnot_batch(state, q, (q + 1) % n_qubits, n_qubits)
    return state


def forward(x_batch, weights, n_qubits=N_QUBITS):
    """weights: (n_layers, n_qubits, 3). Returns expectation <Z0> per sample, shape (batch,)."""
    state = angle_embedding(x_batch, n_qubits)
    for layer in range(weights.shape[0]):
        state = strongly_entangling_layer_batch(state, weights[layer], n_qubits)
    probs = np.abs(state) ** 2  # (batch, DIM)
    # <Z> on qubit 0: +1 if bit0==0 else -1 (qubit 0 is the most-significant reshape axis we used => axis index 0)
    idx = np.arange(DIM)
    bit0 = (idx >> (n_qubits - 1)) & 1  # since axis 0 in our reshape corresponds to qubit0 as the leading axis
    signs = np.where(bit0 == 0, 1.0, -1.0)
    z0 = probs @ signs
    return z0.real


def predict_proba(x_batch, weights):
    z0 = forward(x_batch, weights)
    return (1.0 - z0) / 2.0  # map <Z> in [-1,1] to prob in [0,1]


def bce_loss(y_true, p):
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def param_shift_grad(x_batch, y_batch, weights, shift=np.pi / 2):
    grad = np.zeros_like(weights)
    flat_shape = weights.shape
    it = np.nditer(weights, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        w_plus = weights.copy(); w_plus[idx] += shift
        w_minus = weights.copy(); w_minus[idx] -= shift
        p_plus = predict_proba(x_batch, w_plus)
        p_minus = predict_proba(x_batch, w_minus)
        # dL/dw via chain rule through parameter-shift on the expectation value,
        # combined with BCE gradient w.r.t. p (approximated by finite difference on the loss directly)
        loss_plus = bce_loss(y_batch, p_plus)
        loss_minus = bce_loss(y_batch, p_minus)
        grad[idx] = (loss_plus - loss_minus) / 2.0
    return grad


def train_vqc(X_train, y_train, n_layers=2, n_qubits=N_QUBITS, lr=0.05, epochs=50, batch_size=32, seed=42):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3))
    # Adam state
    m = np.zeros_like(weights); v = np.zeros_like(weights)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X_train.shape[0]
    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        Xs, ys = X_train[perm], y_train[perm]
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size]
            if len(xb) == 0:
                continue
            grad = param_shift_grad(xb, yb, weights)
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            mhat = m / (1 - beta1 ** t)
            vhat = v / (1 - beta2 ** t)
            weights -= lr * mhat / (np.sqrt(vhat) + eps)
        p = predict_proba(X_train, weights)
        history.append(bce_loss(y_train, p))
    return weights, history


def scale_to_angles(X, x_min=None, x_max=None):
    """Scale each of the 4 PCA features to [0, pi] for angle embedding."""
    if x_min is None:
        x_min = X.min(axis=0)
    if x_max is None:
        x_max = X.max(axis=0)
    span = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)
    Xs = (X - x_min) / span * np.pi
    return Xs, x_min, x_max
