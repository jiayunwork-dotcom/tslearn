"""Common benchmark setup: 80 series, length 24, 3 channels (artificial)."""
import numpy as np

rng = np.random.RandomState(0)
n_ts, sz, d = 80, 24, 3
t = np.linspace(0, 4 * np.pi, sz)

X = []
for i in range(n_ts):
    group = i % 3
    wave = np.sin(t + rng.randint(-3, 4))
    channels = [rng.randn(sz) for _ in range(3)]
    channels[group] = wave + 0.1 * rng.randn(sz)
    X.append(np.column_stack(channels))
X = np.array(X)
