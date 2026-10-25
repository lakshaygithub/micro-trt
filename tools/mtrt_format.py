"""
The on-disk format shared by the Python tools and the C++ engine.

A dataset is two files with a common prefix:

  <prefix>.bin   raw little-endian float32 values, back to back
  <prefix>.idx   a text index, one entry per line:

      config <key> <value>                    a scalar (model hyperparameter)
      ints   <key> <v0> <v1> ...               a list of integers (token ids)
      tensor <name> <byte_offset> <rows> <cols>

Every tensor is 2-D and row-major, matching micro-trt's Tensor class; a vector
is stored as 1 x N, the same shape the bias tensors already use.

A text index rather than JSON or safetensors because the C++ side can parse it
with an ifstream and nothing else -- no third-party dependency for the engine.
"""

import numpy as np

# Offsets are aligned so any tensor could be memory-mapped and read with
# vectorised loads without an unaligned first element.
ALIGN = 256


class Writer:
    def __init__(self, prefix):
        self.prefix = prefix
        self.bin = open(prefix + ".bin", "wb")
        self.lines = []
        self.offset = 0

    def config(self, key, value):
        self.lines.append(f"config {key} {value}")

    def ints(self, key, values):
        self.lines.append(f"ints {key} " + " ".join(str(int(v)) for v in values))

    def tensor(self, name, array):
        a = np.ascontiguousarray(array, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.ndim != 2:
            raise ValueError(f"{name}: expected 1-D or 2-D, got shape {a.shape}")

        pad = (-self.offset) % ALIGN
        if pad:
            self.bin.write(b"\0" * pad)
            self.offset += pad

        self.lines.append(f"tensor {name} {self.offset} {a.shape[0]} {a.shape[1]}")
        self.bin.write(a.astype("<f4", copy=False).tobytes())
        self.offset += a.nbytes

    def close(self):
        self.bin.close()
        with open(self.prefix + ".idx", "w") as f:
            f.write("\n".join(self.lines) + "\n")
