"""
Neural Network — Obstacle / Walkable Cell Classifier
------------------------------------------------------
Real-world use: Aerial image segmentation (Google Maps satellite),
                Self-driving car road detection,
                Medical image analysis (tumor vs. normal),
                Building footprint detection from drones

AI Category: Supervised Machine Learning (Binary Classification)

Architecture: 2-layer Feedforward Neural Network (from scratch)
  Input  layer: 9 features (3×3 pixel patch brightness)
  Hidden layer: 16 neurons, ReLU activation
  Output layer: 1 neuron, Sigmoid activation → P(walkable)

Why Neural Network?
  Rule-based threshold (pixel > 200 = white = walkable) fails on:
  - Grayscale floors that aren't pure white
  - Shadows on walkable paths
  - Anti-aliased wall edges
  The NN LEARNS a more robust boundary from examples.

Training: Supervised — generate labels from known threshold,
          train NN to generalize beyond that threshold.
          Loss: Binary Cross-Entropy
          Optimizer: SGD with manual backpropagation

NO external ML libraries (numpy only for matrix math).
Full forward pass, backprop, weight update implemented manually.
"""

import math
import random


# ── Math Primitives (no numpy for core logic) ────────────────────────────

def sigmoid(x):
    """σ(x) = 1 / (1 + e^-x)   Maps any real → (0,1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def relu(x):
    """ReLU = max(0, x)   Avoids vanishing gradient, fast to compute."""
    return max(0.0, x)


def relu_deriv(x):
    """Derivative of ReLU: 1 if x>0 else 0."""
    return 1.0 if x > 0 else 0.0


def dot(v1, v2):
    """Dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def bce_loss(y_true, y_pred, eps=1e-7):
    """Binary Cross Entropy: -[y*log(p) + (1-y)*log(1-p)]"""
    y_pred = max(eps, min(1 - eps, y_pred))
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


# ── Neural Network ───────────────────────────────────────────────────────

class NeuralNetClassifier:
    """
    2-layer NN for binary classification: walkable (1) vs obstacle (0).

    Architecture:
        Input (9) → [W1, b1] → Hidden (16, ReLU) → [W2, b2] → Output (1, Sigmoid)

    All weights initialized with Xavier/Glorot initialization:
        W ~ Uniform(-sqrt(6/(n_in+n_out)), +sqrt(6/(n_in+n_out)))
    """

    def __init__(self, input_size=9, hidden_size=16, lr=0.05, seed=42):
        random.seed(seed)
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.lr = lr

        # Xavier initialization
        limit1 = math.sqrt(6.0 / (input_size + hidden_size))
        limit2 = math.sqrt(6.0 / (hidden_size + 1))

        # W1: hidden_size × input_size
        self.W1 = [[random.uniform(-limit1, limit1) for _ in range(input_size)]
                   for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size

        # W2: 1 × hidden_size
        self.W2 = [random.uniform(-limit2, limit2) for _ in range(hidden_size)]
        self.b2 = 0.0

        self.train_losses = []
        self.epochs_trained = 0

    # ── Forward Pass ─────────────────────────────────────────────────────

    def forward(self, x):
        """
        x: list of input_size floats (normalized 0-1)
        Returns: (output probability, hidden activations, hidden pre-activations)
        """
        # Hidden layer
        z1 = [dot(self.W1[j], x) + self.b1[j] for j in range(self.hidden_size)]
        a1 = [relu(z) for z in z1]

        # Output layer
        z2 = dot(self.W2, a1) + self.b2
        a2 = sigmoid(z2)

        return a2, a1, z1

    # ── Backward Pass (Manual Backpropagation) ────────────────────────────

    def backward(self, x, y_true, a2, a1, z1):
        """
        Compute gradients via chain rule, update weights.
        Chain rule:
          dL/dW2 = dL/da2 * da2/dz2 * dz2/dW2
          dL/dW1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dW1
        """
        # Output layer gradient
        # d(BCE)/d(a2) * d(sigmoid)/d(z2) = (a2 - y_true)  [BCE + sigmoid simplifies]
        delta2 = a2 - y_true

        # Gradients for W2, b2
        dW2 = [delta2 * a1[j] for j in range(self.hidden_size)]
        db2 = delta2

        # Hidden layer gradient (backprop through ReLU)
        delta1 = [self.W2[j] * delta2 * relu_deriv(z1[j])
                  for j in range(self.hidden_size)]

        # Gradients for W1, b1
        dW1 = [[delta1[j] * x[i] for i in range(self.input_size)]
               for j in range(self.hidden_size)]
        db1 = delta1[:]

        # SGD Update: θ ← θ - α * dθ
        for j in range(self.hidden_size):
            for i in range(self.input_size):
                self.W1[j][i] -= self.lr * dW1[j][i]
            self.b1[j] -= self.lr * db1[j]

        for j in range(self.hidden_size):
            self.W2[j] -= self.lr * dW2[j]
        self.b2 -= self.lr * db2

    # ── Training ─────────────────────────────────────────────────────────

    def train(self, X, y, epochs=30):
        """
        X: list of samples (each = list of input_size floats)
        y: list of labels (0 or 1)
        epochs: full passes through data
        """
        indices = list(range(len(X)))
        for epoch in range(epochs):
            random.shuffle(indices)
            epoch_loss = 0.0
            for i in indices:
                a2, a1, z1 = self.forward(X[i])
                loss = bce_loss(y[i], a2)
                epoch_loss += loss
                self.backward(X[i], y[i], a2, a1, z1)
            avg_loss = epoch_loss / len(X)
            self.train_losses.append(avg_loss)
            self.epochs_trained += 1

    def predict(self, x):
        """Returns P(walkable) for a single sample."""
        prob, _, _ = self.forward(x)
        return prob

    def predict_binary(self, x, threshold=0.5):
        """Returns 1 (walkable) or 0 (obstacle)."""
        return 1 if self.predict(x) >= threshold else 0


# ── Feature Extraction ────────────────────────────────────────────────────

def extract_patch_features(image_gray, r, c, patch_size=3):
    """
    Extract a patch_size×patch_size neighborhood centered at (r,c).
    Normalize to [0, 1].
    Returns a flat list of patch_size² floats.
    """
    half = patch_size // 2
    h, w = len(image_gray), len(image_gray[0])
    features = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            nr = max(0, min(h - 1, r + dr))
            nc = max(0, min(w - 1, c + dc))
            features.append(image_gray[nr][nc] / 255.0)
    return features


def build_training_data(image_gray, threshold=200, sample_rate=0.05):
    """
    Generate training data from image using threshold as ground truth label.
    Samples sample_rate fraction of pixels to keep training fast.
    Returns (X, y, class_balance).
    """
    h, w = len(image_gray), len(image_gray[0])
    X, y = [], []
    pos, neg = 0, 0

    for r in range(h):
        for c in range(w):
            if random.random() > sample_rate:
                continue
            features = extract_patch_features(image_gray, r, c)
            label = 1 if image_gray[r][c] >= threshold else 0
            X.append(features)
            y.append(label)
            if label == 1:
                pos += 1
            else:
                neg += 1

    return X, y, (pos, neg)


def nn_predict_grid(nn_model, image_gray, threshold=0.5):
    """
    Apply trained NN to every pixel, return binary grid.
    1 = walkable, 0 = obstacle.

    🔧 FIX: Added post-processing to clean up noise and ensure connectivity
    """
    h, w = len(image_gray), len(image_gray[0])
    grid = [[0] * w for _ in range(h)]

    # First pass: raw prediction
    for r in range(h):
        for c in range(w):
            features = extract_patch_features(image_gray, r, c)
            grid[r][c] = nn_model.predict_binary(features, threshold)

    # 🔧 FIX: Post-processing - remove isolated pixels (noise removal)
    # This helps clean up the grid and improve A* performance
    cleaned = [row[:] for row in grid]
    for r in range(1, h-1):
        for c in range(1, w-1):
            # Count walkable neighbors
            neighbors = sum(grid[nr][nc] for nr in range(r-1, r+2) 
                          for nc in range(c-1, c+2) if (nr, nc) != (r, c))
            # If isolated walkable pixel surrounded by obstacles, make it obstacle
            if grid[r][c] == 1 and neighbors <= 1:
                cleaned[r][c] = 0
            # If isolated obstacle surrounded by walkable, make it walkable
            elif grid[r][c] == 0 and neighbors >= 7:
                cleaned[r][c] = 1

    return cleaned