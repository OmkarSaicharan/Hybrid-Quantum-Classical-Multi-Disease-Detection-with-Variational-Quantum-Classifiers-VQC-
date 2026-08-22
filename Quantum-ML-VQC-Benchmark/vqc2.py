import numpy as np


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128)


def rot(phi, theta, omega):
    return rz(omega) @ ry(theta) @ rz(phi)


def apply_single_qubit_gate_batch(state, gate, qubit, n_qubits):
    batch = state.shape[0]
    shape = (batch,) + (2,) * n_qubits
    st = state.reshape(shape)
    st = np.moveaxis(st, qubit + 1, 1)
    flat = st.reshape(batch, 2, -1)
    flat = np.einsum('ij,bjk->bik', gate, flat)
    st = flat.reshape(st.shape)
    st = np.moveaxis(st, 1, qubit + 1)
    return st.reshape(batch, -1)


def apply_cnot_batch(state, control, target, n_qubits):
    batch = state.shape[0]
    shape = (batch,) + (2,) * n_qubits
    st = state.reshape(shape)
    st = np.moveaxis(st, [control + 1, target + 1], [1, 2])
    out = st.copy()
    idx1 = [slice(None)] * st.ndim
    idx1[1] = 1
    sub = st[tuple(idx1)]
    sub_flipped = np.flip(sub, axis=1)
    out[tuple(idx1)] = sub_flipped
    out = np.moveaxis(out, [1, 2], [control + 1, target + 1])
    return out.reshape(batch, -1)


def angle_embedding(x_batch, n_qubits):
    batch = x_batch.shape[0]
    dim = 2 ** n_qubits
    state = np.zeros((batch, dim), dtype=np.complex128)
    state[:, 0] = 1.0
    for q in range(n_qubits):
        g_angles = x_batch[:, q]
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


def apply_depolarizing_noise_batch(state, qubit, n_qubits, p, rng):
    """Monte-Carlo single-qubit depolarizing channel: with prob p, apply a uniformly random Pauli (X, Y, Z)."""
    if p <= 0:
        return state
    batch = state.shape[0]
    draws = rng.random(batch)
    do_noise = draws < p
    if not do_noise.any():
        return state
    which = rng.integers(0, 3, size=batch)  # 0=X,1=Y,2=Z
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    paulis = [X, Y, Z]
    new_state = state.copy()
    for p_idx in range(3):
        mask = do_noise & (which == p_idx)
        if not mask.any():
            continue
        sub = state[mask]
        sub_noised = apply_single_qubit_gate_batch(sub, paulis[p_idx], qubit, n_qubits)
        new_state[mask] = sub_noised
    return new_state


def strongly_entangling_layer_batch(state, params_layer, n_qubits, entangle=True, noise_p=0.0, rng=None):
    for q in range(n_qubits):
        phi, theta, omega = params_layer[q]
        state = apply_single_qubit_gate_batch(state, rot(phi, theta, omega), q, n_qubits)
        if noise_p > 0:
            state = apply_depolarizing_noise_batch(state, q, n_qubits, noise_p, rng)
    if entangle:
        for q in range(n_qubits):
            state = apply_cnot_batch(state, q, (q + 1) % n_qubits, n_qubits)
            if noise_p > 0:
                state = apply_depolarizing_noise_batch(state, q, n_qubits, noise_p, rng)
                state = apply_depolarizing_noise_batch(state, (q + 1) % n_qubits, n_qubits, noise_p, rng)
    return state


def forward(x_batch, weights, n_qubits, entangle=True, noise_p=0.0, seed=None):
    dim = 2 ** n_qubits
    state = angle_embedding(x_batch, n_qubits)
    rng = np.random.default_rng(seed) if noise_p > 0 else None
    for layer in range(weights.shape[0]):
        state = strongly_entangling_layer_batch(state, weights[layer], n_qubits, entangle=entangle,
                                                 noise_p=noise_p, rng=rng)
    probs = np.abs(state) ** 2
    idx = np.arange(dim)
    bit0 = (idx >> (n_qubits - 1)) & 1
    signs = np.where(bit0 == 0, 1.0, -1.0)
    z0 = probs @ signs
    return z0.real


def predict_proba(x_batch, weights, n_qubits, entangle=True, noise_p=0.0, seed=None):
    z0 = forward(x_batch, weights, n_qubits, entangle=entangle, noise_p=noise_p, seed=seed)
    return (1.0 - z0) / 2.0


def bce_loss(y_true, p):
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def param_shift_grad(x_batch, y_batch, weights, n_qubits, entangle=True, shift=np.pi / 2):
    grad = np.zeros_like(weights)
    it = np.nditer(weights, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        w_plus = weights.copy(); w_plus[idx] += shift
        w_minus = weights.copy(); w_minus[idx] -= shift
        p_plus = predict_proba(x_batch, w_plus, n_qubits, entangle=entangle)
        p_minus = predict_proba(x_batch, w_minus, n_qubits, entangle=entangle)
        loss_plus = bce_loss(y_batch, p_plus)
        loss_minus = bce_loss(y_batch, p_minus)
        grad[idx] = (loss_plus - loss_minus) / 2.0
    return grad


def train_vqc(X_train, y_train, n_layers=2, n_qubits=4, entangle=True, lr=0.05, epochs=40, batch_size=32, seed=42):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3))
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
            grad = param_shift_grad(xb, yb, weights, n_qubits, entangle=entangle)
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            mhat = m / (1 - beta1 ** t)
            vhat = v / (1 - beta2 ** t)
            weights -= lr * mhat / (np.sqrt(vhat) + eps)
        p = predict_proba(X_train, weights, n_qubits, entangle=entangle)
        history.append(bce_loss(y_train, p))
    return weights, history


def scale_to_angles(X, x_min=None, x_max=None):
    if x_min is None:
        x_min = X.min(axis=0)
    if x_max is None:
        x_max = X.max(axis=0)
    span = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)
    Xs = (X - x_min) / span * np.pi
    return Xs, x_min, x_max


def quantum_kernel_matrix(X1, X2, n_qubits):
    """Fidelity-based quantum kernel: K(x,y) = |<phi(x)|phi(y)>|^2 using angle-embedded states (no trainable layers)."""
    s1 = angle_embedding(X1, n_qubits)   # (n1, dim) complex
    s2 = angle_embedding(X2, n_qubits)   # (n2, dim) complex
    overlap = s1 @ s2.conj().T           # (n1, n2) complex
    return np.abs(overlap) ** 2
