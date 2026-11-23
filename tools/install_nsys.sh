#!/usr/bin/env bash
# Makes sure Nsight Systems' command-line profiler, nsys, is installed.
#
# Colab's image does not always include it. This installs it from NVIDIA's CUDA
# apt repository, matching the CUDA version nvcc reports, and links the binary
# onto the PATH. Run it as root (Colab cells already are):
#
#     !bash tools/install_nsys.sh
set -euo pipefail

if command -v nsys >/dev/null 2>&1; then
    nsys --version
    exit 0
fi

# "release 12.4" -> "12-4", the suffix NVIDIA's package names use.
cuda_ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')

apt-get update -qq
if ! apt-get install -y -qq "cuda-nsight-systems-${cuda_ver}"; then
    if ! apt-get install -y -qq nsight-systems-cli; then
        echo "Could not install Nsight Systems. Packages this machine can see:" >&2
        apt-cache search nsight-systems >&2 || true
        exit 1
    fi
fi

# The packages install under /opt/nvidia or the CUDA directory; put nsys on the
# PATH so the Makefile can call it by name.
if ! command -v nsys >/dev/null 2>&1; then
    nsys_bin=$(find /opt/nvidia /usr/local -path '*/bin/nsys' 2>/dev/null | head -n 1 || true)
    if [ -z "${nsys_bin}" ]; then
        echo "Installed, but no nsys binary was found under /opt/nvidia or /usr/local" >&2
        exit 1
    fi
    ln -sf "${nsys_bin}" /usr/local/bin/nsys
fi
nsys --version
