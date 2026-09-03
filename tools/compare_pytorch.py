#!/usr/bin/env python3
"""
Verify micro-trt's MLP output against PyTorch, and time the same network.

Reads the binary dump written by bench_mlp, rebuilds the identical network in
PyTorch with the identical weights, and reports two things:

  1. Whether the outputs agree numerically. This is an independent check --
     PyTorch shares no code with micro-trt, so agreement is real evidence
     rather than two implementations making the same mistake.

  2. How long PyTorch takes for the same forward pass on the same GPU.

Usage:  python3 tools/compare_pytorch.py [path/to/mlp_dump.bin]
"""

import struct
import sys
import time

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed. Run: pip install torch")


def read_dump(path):
    """Parse the binary written by MLP::dump."""
    with open(path, "rb") as f:
        blob = f.read()

    off = 0

    def take_i32(n=1):
        nonlocal off
        vals = struct.unpack_from(f"<{n}i", blob, off)
        off += 4 * n
        return vals

    def take_f32(n):
        nonlocal off
        arr = np.frombuffer(blob, dtype=np.float32, count=n, offset=off)
        off += 4 * n
        return arr

    (n_layers,) = take_i32()
    (batch,) = take_i32()

    shapes = []
    for _ in range(n_layers):
        in_f, out_f, relu = take_i32(3)
        shapes.append((in_f, out_f, bool(relu)))

    x = take_f32(batch * shapes[0][0]).reshape(batch, shapes[0][0])

    weights = []
    for in_f, out_f, _ in shapes:
        w = take_f32(in_f * out_f).reshape(in_f, out_f)
        b = take_f32(out_f)
        weights.append((w, b))

    out_dim = shapes[-1][1]
    y = take_f32(batch * out_dim).reshape(batch, out_dim)

    if off != len(blob):
        sys.exit(f"dump file has {len(blob) - off} unexpected trailing bytes")

    return batch, shapes, x, weights, y


def build_forward(shapes, weights, device):
    """Return a closure running the same network in PyTorch."""
    tensors = []
    for (w, b), (_, _, relu) in zip(weights, shapes):
        # micro-trt stores W as (in x out) and computes X @ W.
        # torch.nn.functional.linear expects (out x in) and computes X @ W.T,
        # so the weight is transposed here. Getting this wrong is the single
        # most likely source of a false mismatch.
        tw = torch.from_numpy(np.ascontiguousarray(w.T)).to(device)
        tb = torch.from_numpy(np.ascontiguousarray(b)).to(device)
        tensors.append((tw, tb, relu))

    def forward(x):
        h = x
        for tw, tb, relu in tensors:
            h = torch.nn.functional.linear(h, tw, tb)
            if relu:
                h = torch.relu(h)
        return torch.softmax(h, dim=1)

    return forward


def time_forward(forward, x, iters=50, reps=15, warmup_seconds=1.5):
    """Time with CUDA events and report the median, mirroring the C++ harness."""
    deadline = time.time() + warmup_seconds
    while time.time() < deadline:
        for _ in range(20):
            forward(x)
        torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        forward(x)
        torch.cuda.synchronize()

        start.record()
        for _ in range(iters):
            forward(x)
        stop.record()
        torch.cuda.synchronize()

        samples.append(start.elapsed_time(stop) / iters)

    samples.sort()
    return samples[len(samples) // 2], samples[0], samples[-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "mlp_dump.bin"

    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible to PyTorch.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    batch, shapes, x_np, weights, ours_np = read_dump(path)

    arch = " -> ".join([str(shapes[0][0])] + [str(s[1]) for s in shapes])
    print(f"Network: {arch}   (batch {batch})")

    flops = sum(2.0 * batch * i * o for i, o, _ in shapes)
    print(f"Arithmetic: {flops / 1e9:.2f} GFLOP per forward pass\n")

    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    forward = build_forward(shapes, weights, device)

    with torch.no_grad():
        theirs = forward(x).cpu().numpy()

        # ---- correctness ----
        ours = ours_np
        diff = np.abs(ours - theirs)
        scale = np.maximum(np.abs(theirs), 1e-6)
        rel = diff / scale
        worst = rel.max()

        row_sums = ours.sum(axis=1)
        row_err = np.abs(row_sums - 1.0).max()

        print("Correctness vs PyTorch")
        print(f"  worst relative difference : {worst:.3g}")
        print(f"  worst absolute difference : {diff.max():.3g}")
        print(f"  our rows sum to 1 within  : {row_err:.3g}")
        agree = worst < 5e-3
        print(f"  agreement                 : {'yes' if agree else 'NO'}\n")

        # ---- timing ----
        median, lo, hi = time_forward(forward, x)

    spread = 100.0 * (hi - lo) / median
    print("PyTorch forward pass")
    print(f"  median         : {median:.4f} ms  "
          f"(min {lo:.4f}, max {hi:.4f}, spread {spread:.1f}%)")
    print(f"  throughput     : {flops / (median / 1000.0) / 1e9:.1f} GFLOP/s")
    print(f"  per sample     : {median * 1000.0 / batch:.2f} us")
    print(f"  samples/second : {batch / (median / 1000.0):.0f}\n")

    print("Compare the median above against the micro-trt figure printed by "
          "bench_mlp.")
    print("Note PyTorch dispatches to cuBLAS and cuDNN, which are hand-tuned "
          "in assembly\nby NVIDIA. Matching them is not the goal; knowing the "
          "gap is.")

    sys.exit(0 if agree else 1)


if __name__ == "__main__":
    main()