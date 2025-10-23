import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates over nqubits wires."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(angles):
    """Apply RY rotations with per-qubit angles.

    Supports batched angles with shape [batch, n_qubits]: in this case, each
    wire receives a vector of angles of length 'batch', leveraging PennyLane's
    built-in parameter broadcasting to evaluate the batch in a single QNode call.
    """
    if hasattr(angles, "ndim") and angles.ndim == 2:
        # angles shape: [batch, n_qubits]
        n_qubits = angles.shape[1]
        for idx in range(n_qubits):
            qml.RY(angles[:, idx], wires=idx)
    else:
        for idx, angle in enumerate(angles):
            qml.RY(angle, wires=idx)


def entangling_layer(nqubits):
    """Apply a brickwork pattern of CNOTs (even then odd pairs)."""
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def create_Hermitian(N, A, B, D):
    """Construct a 2^n_local x 2^n_local Hermitian from parameter vectors.

    The layout matches the original ANO construction: D fills the diagonal;
    A and B parameterize the strict lower-triangular part (real and imag).
    """
    h = torch.zeros((N, N), dtype=torch.complex128, device=D.device)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()
        count += i
    H = h + h.conj().T
    return H


def build_quantum_net(n_qubits, vqc_depth, n_local):
    """Factory that returns a QNode-callable circuit with fixed hyperparameters."""

    def quantum_net(X, theta, H_bank):
        H_layer(n_qubits)
        RY_layer(X)
        for k in range(vqc_depth):
            entangling_layer(n_qubits)
            RY_layer(theta[k])
        # Sliding window of size n_local starting at each qubit index (wrap-around)
        exp_vals = [
            qml.expval(
                qml.Hermitian(
                    H_bank[q],
                    wires=(np.arange(q, q + n_local) % n_qubits).tolist(),
                )
            )
            for q in range(n_qubits)
        ]
        return exp_vals

    return quantum_net


class ANO_VQC_Model(nn.Module):
    """Adaptive Nonlocal Observable VQC head for ABCDisCo.

    Configurable parameters:
      - n_qubits: number of wires and input features consumed per sample
      - n_outputs: number of class logits returned (first n_outputs measurements)
      - n_local: locality (window size) for each learned observable
      - vqc_depth: number of entangling+rotation blocks
      - qdevice: PennyLane device string (e.g., "default.qubit", "lightning.qubit", "lightning.gpu")

    Notes:
      - Inputs are reshaped to (-1, n_qubits). If feature count != n_qubits, this will error.
      - If qdevice is CPU-only, tensors are moved CPU->QNode and back; if qdevice is a Lightning GPU
        backend, tensors/parameters are moved to CUDA for the QNode call.
    """

    def __init__(self,
                 n_qubits: int = 11,
                 n_outputs: int = 2,
                 n_local: int = 1,
                 vqc_depth: int = 4,
                 qdevice: str = "default.qubit"):
        super(ANO_VQC_Model, self).__init__()
        self.n_qubits = int(n_qubits)
        self.n_outputs = int(n_outputs)
        self.n_local = int(n_local)
        self.vqc_depth = int(vqc_depth)
        self.qdevice = str(qdevice)

        # Torch device used inside the QNode call
        use_gpu_backend = self.qdevice in ("lightning.gpu", "lightning.qubit")
        self.qnode_torch_device = torch.device("cuda" if (use_gpu_backend and torch.cuda.is_available()) else "cpu")

        # Trainable VQC rotation parameters (depth x n_qubits)
        self.theta = nn.Parameter(0.01 * torch.randn(self.vqc_depth, self.n_qubits))

        # Trainable Hermitian parameters per position (window size = n_local)
        self.N = 2 ** self.n_local
        self.A = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in range(self.n_qubits)])
        self.B = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in range(self.n_qubits)])
        self.D = nn.ParameterList([nn.Parameter(torch.empty(self.N)) for _ in range(self.n_qubits)])
        for q in range(self.n_qubits):
            nn.init.normal_(self.A[q], std=2.)
            nn.init.normal_(self.B[q], std=2.)
            nn.init.normal_(self.D[q], std=2.)

        # PennyLane device and QNode
        self.dev = qml.device(self.qdevice, wires=self.n_qubits)
        qfunc = build_quantum_net(self.n_qubits, self.vqc_depth, self.n_local)
        self.VQC = qml.QNode(qfunc, self.dev, interface="torch")

    def _build_observable_bank(self, target_device: torch.device):
        """Create the list of Hermitian observables on target_device for current parameters."""
        H_bank = [
            create_Hermitian(self.N,
                             self.A[q].to(target_device),
                             self.B[q].to(target_device),
                             self.D[q].to(target_device))
            for q in range(self.n_qubits)
        ]
        return H_bank

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Remember caller's device to return logits there
        caller_device = X.device

        # Reshape inputs into per-qubit angles
        z1 = X.reshape(-1, self.n_qubits)

        # Select devices for QNode execution
        z1_q = z1.to(self.qnode_torch_device)
        theta_q = self.theta.to(self.qnode_torch_device)
        H_bank = self._build_observable_bank(self.qnode_torch_device)

        # Evaluate QNode once per batch (leveraging parameter broadcasting)
        q_out = self.VQC(z1_q, theta_q, H_bank)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack([torch.as_tensor(t) for t in q_out], dim=-1)
        q_out = torch.as_tensor(q_out, dtype=torch.float32)

        # Class logits are the first n_outputs measurement heads
        logits = q_out[:, : self.n_outputs].to(caller_device)
        return logits


def build_hybrid_qnet(n_qubits: int, ent_layers: int):
    """Factory for a hybrid QNode using AngleEmbedding + StronglyEntanglingLayers."""

    wires = list(range(n_qubits))

    def qfunc(angles, weights):
        qml.AngleEmbedding(angles, wires=wires, rotation="Y")
        qml.StronglyEntanglingLayers(weights=weights, wires=wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return qfunc


class HybridQNN(nn.Module):
    """Simple hybrid model: FCN(11→hidden→n_qubits) → VQC → FCN(n_qubits→n_outputs).

    Config:
      - n_features: input feature dimension (default 11)
      - hidden_dim: size of first hidden layer (default 64)
      - n_qubits: number of qubits/angles (default 6)
      - vqc_depth: number of StronglyEntanglingLayers (default 4)
      - n_outputs: output logits (default 2)
      - qdevice: PennyLane device string (default "default.qubit")
    """

    def __init__(self,
                 n_features: int = 11,
                 hidden_dim: int = 64,
                 n_qubits: int = 6,
                 vqc_depth: int = 4,
                 n_outputs: int = 2,
                 qdevice: str = "default.qubit"):
        super(HybridQNN, self).__init__()
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.n_qubits = int(n_qubits)
        self.vqc_depth = int(vqc_depth)
        self.n_outputs = int(n_outputs)
        self.qdevice = str(qdevice)

        use_gpu_backend = self.qdevice in ("lightning.gpu", "lightning.qubit")
        self.qnode_torch_device = torch.device("cuda" if (use_gpu_backend and torch.cuda.is_available()) else "cpu")

        # Classical front FCN with BatchNorm then map to qubit angles
        self.fc1 = nn.Linear(self.n_features, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_qubits)

        # Quantum template parameters for StronglyEntanglingLayers
        # Expected shape: (layers, n_wires, 3)
        self.weights = nn.Parameter(0.01 * torch.randn(self.vqc_depth, self.n_qubits, 3))

        # Device and QNode
        self.dev = qml.device(self.qdevice, wires=self.n_qubits)
        qfunc = build_hybrid_qnet(self.n_qubits, self.vqc_depth)
        self.QNODE = qml.QNode(qfunc, self.dev, interface="torch")

        # Classical back FCN to logits
        self.head = nn.Linear(self.n_qubits, self.n_outputs)

    def _scale_to_angles(self, x: torch.Tensor) -> torch.Tensor:
        # Optionally scale to [-pi, pi] via tanh
        return torch.tanh(x) * np.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        caller_device = x.device
        # Front FCN with BN+ReLU once
        h = self.fc1(x)
        h = self.bn1(h)
        h = torch.relu(h)
        angles = self.fc2(h)
        angles = self._scale_to_angles(angles)

        # Move inputs/params to the QNode device
        angles_q = angles.to(self.qnode_torch_device)
        weights_q = self.weights.to(self.qnode_torch_device)

        # Evaluate QNode in a single batched call → [batch, n_qubits]
        q_out = self.QNODE(angles_q, weights_q)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack([torch.as_tensor(t) for t in q_out], dim=-1)
        q_out = torch.as_tensor(q_out, dtype=torch.float32)

        # Back FCN to logits on caller device
        logits = self.head(q_out.to(caller_device))
        return logits

