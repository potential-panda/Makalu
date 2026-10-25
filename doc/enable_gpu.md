# LightGBM AMD GPU Build Fixes (Garuda/Arch Linux)

This document summarizes the manual fixes required to enable AMD GPU (OpenCL) support for LightGBM on systems running **Boost 1.89+** and **AMD RDNA 3 (gfx1151)** hardware.

## 1. Environment & Prerequisites
- **OS:** Garuda Linux (Arch-based)
- **GPU:** AMD Radeon (Integrated/Discrete)
- **Compiler:** GCC 15.2+ / CMake 4.2+
- **Key Conflict:** Boost 1.89 removed `libboost_system.so`, causing standard CMake scripts to fail.

## 2. Dependencies
Ensure the ROCm stack and OpenCL headers are installed:

```bash
sudo pacman -S boost boost-libs rocm-opencl-runtime clinfo opencl-headers
```

## 3. Manual Source Fixes

### A. Fix Boost 1.89 Incompatibility (CMakeLists.txt)
Modify the root CMakeLists.txt (around line 195) to stop searching for the non-existent system library and point to headers manually.

Change:

```CMake
find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
```

To:

```CMake
# 'system' is now header-only in Boost 1.89+
find_package(Boost 1.56.0 COMPONENTS filesystem REQUIRED)
set(Boost_INCLUDE_DIRS "/usr/include")
```

### B. Fix SHA1 Type Conversion Error
Boost 1.89 tightened type safety for UUID/SHA1. The internal boost::compute headers in LightGBM must be patched.

File: external_libs/compute/include/boost/compute/detail/sha1.hpp
Line 41 Change:

```C++
// Change this:
h.get_digest(digest);

// To this:
h.get_digest(reinterpret_cast<unsigned char (&)[20]>(digest));
```

## 4. Build Process
Run a clean build from the build directory:

```Bash
rm -rf *
cmake -DUSE_GPU=1 ..
make -j$(nproc)
```

## 5. Python Integration (Hot-Swap Method)
Since pip often fails to find the root CMakeLists.txt during sub-package builds, use the manual linking method:

Copy the compiled binary:

```Bash
cp lib_lightgbm.so python-package/lightgbm/
```

Manually copy the binary to your active site-packages:

```Bash
# Adjust path to your specific pyenv/python version
cp lib_lightgbm.so ~/.pyenv/versions/3.13.9/lib/python3.13/site-packages/lightgbm/
```

## 6. Verification
Run the following Python snippet to confirm the AMD GPU is active:

```Python
import lightgbm as lgb

params = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
# Look for: [LightGBM] [Info] Using GPU Device: gfx1151
```

