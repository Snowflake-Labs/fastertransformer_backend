import re

path = "_deps/repo-ft-src/src/fastertransformer/utils/cuda_type_utils.cuh"
with open(path) as f:
    src = f.read()

# Replace ambiguous __habs(half) -> explicit cast
src = re.sub(
    r"(template<> __device__ inline half\s+cuda_abs\(half\s+val\)\s*\{)\s*return __habs\(val\);",
    r"\1 return __habs((__half)val);", src)
# Replace ambiguous __habs2(half2) -> explicit cast
src = re.sub(
    r"(template<> __device__ inline half2\s+cuda_abs\(half2\s+val\)\s*\{)\s*return __habs2\(val\);",
    r"\1 return __habs2((__half2)val);", src)
# Replace ambiguous __habs(__nv_bfloat16) -> float abs
src = re.sub(
    r"(template<> __device__ inline __nv_bfloat16\s+cuda_abs\(__nv_bfloat16\s+val\)\s*\{)\s*return __habs\(val\);",
    r"\1 return __nv_bfloat16(__builtin_fabsf(float(val)));", src)
# Replace ambiguous __habs2(__nv_bfloat162) -> float abs
src = re.sub(
    r"(template<> __device__ inline __nv_bfloat162\s+cuda_abs\(__nv_bfloat162\s+val\)\s*\{)\s*return __habs2\(val\);",
    r"\1 return __nv_bfloat162(__builtin_fabsf(float(val.x)), __builtin_fabsf(float(val.y)));", src)
# Replace fabs on bfloat16 in else-branch
src = re.sub(
    r"return fabs\(val\);",
    r"return __nv_bfloat16(__builtin_fabsf(float(val)));", src)
src = re.sub(
    r"return make_bfloat162\(fabs\(val\.x\), fabs\(val\.y\)\);",
    r"return __nv_bfloat162(__builtin_fabsf(float(val.x)), __builtin_fabsf(float(val.y)));", src)

with open(path, "w") as f:
    f.write(src)

print("Patched cuda_type_utils.cuh OK")

# ---------------------------------------------------------------------------
# Patch cuda_bf16_fallbacks.cuh — CUDA 13+ provides operator+, operator*,
# and make_bfloat162 for __nv_bfloat162 unconditionally (all archs), so FT's
# fallback definitions for __CUDA_ARCH__ < 800 now cause ambiguous overloads.
# Gate them out when compiling with CUDA 13+.
# ---------------------------------------------------------------------------
bf16_path = "_deps/repo-ft-src/src/fastertransformer/utils/cuda_bf16_fallbacks.cuh"
with open(bf16_path) as f:
    bf16_src = f.read()

bf16_src = bf16_src.replace(
    "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)\n"
    "inline __device__ __nv_bfloat162 operator*",
    "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800) "
    "&& !(defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13)\n"
    "inline __device__ __nv_bfloat162 operator*",
    1,
)

with open(bf16_path, "w") as f:
    f.write(bf16_src)

print("Patched cuda_bf16_fallbacks.cuh OK")

# ---------------------------------------------------------------------------
# Patch cublasMMWrapper.h — GCC 13 no longer transitively includes <array>.
# ---------------------------------------------------------------------------
cublas_h_path = "_deps/repo-ft-src/src/fastertransformer/utils/cublasMMWrapper.h"
with open(cublas_h_path) as f:
    cublas_h = f.read()

cublas_h = cublas_h.replace(
    '#include <map>\n',
    '#include <array>\n#include <map>\n',
    1,
)

with open(cublas_h_path, "w") as f:
    f.write(cublas_h)

print("Patched cublasMMWrapper.h OK")

# ---------------------------------------------------------------------------
# Patch cub::Max/Sum — In CUDA 13.1 (CCCL), these old CUB functors were
# removed.  cub::Max -> ::cuda::maximum<>, cub::Sum -> ::cuda::std::plus<>.
# ---------------------------------------------------------------------------
import glob, os

ft_root = "_deps/repo-ft-src"
cub_files = (
    glob.glob(os.path.join(ft_root, "**", "*.cu"), recursive=True)
    + glob.glob(os.path.join(ft_root, "**", "*.cuh"), recursive=True)
)

patched_cub = []
for fp in cub_files:
    with open(fp) as f:
        text = f.read()
    orig = text
    text = text.replace("cub::Max()", "::cuda::maximum<>{}")
    text = text.replace("cub::Sum()", "::cuda::std::plus<>{}")
    text = text.replace("cub::Sum ", "::cuda::std::plus<> ")
    if text != orig:
        with open(fp, "w") as f:
            f.write(text)
        patched_cub.append(fp)

print(f"Patched cub::Max/Sum in {len(patched_cub)} files: {patched_cub}")

# ---------------------------------------------------------------------------
# Patch utils/CMakeLists.txt — nvToolsExt was removed in CUDA 13.1.  The
# nvtx_utils target unconditionally links -lnvToolsExt; gate it on USE_NVTX.
# ---------------------------------------------------------------------------
utils_cmake = os.path.join(ft_root, "src/fastertransformer/utils/CMakeLists.txt")
with open(utils_cmake) as f:
    ucm = f.read()

ucm = ucm.replace(
    "target_link_libraries(nvtx_utils PUBLIC -lnvToolsExt)",
    "if(USE_NVTX)\n"
    "  target_link_libraries(nvtx_utils PUBLIC -lnvToolsExt)\n"
    "endif()",
    1,
)

with open(utils_cmake, "w") as f:
    f.write(ucm)

print("Patched utils/CMakeLists.txt OK")
