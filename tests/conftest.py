import os

# Force JAX onto CPU for tests. Wavelet computations are small enough that CPU
# is fast, and pytest-xdist workers each try to grab the full GPU under the
# default GPU backend, which causes CUDA_OUT_OF_MEMORY when -n > 1.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
