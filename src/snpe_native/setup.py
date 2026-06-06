import os
from setuptools import setup, Extension

try:
    import pybind11
except ImportError:
    raise RuntimeError("pybind11 is required to build this extension. Please 'pip install pybind11' first.")

snpe_root = os.environ.get('SNPE_ROOT')
if not snpe_root:
    raise RuntimeError("SNPE_ROOT environment variable must be set to build snpe_native")

include_dirs = [
    pybind11.get_include(),
    os.path.join(snpe_root, 'include', 'zdl'),
    os.path.join(snpe_root, 'include')
]

# For aarch64 on Rubik Pi 3, the library is usually at 'lib/aarch64-oe-linux-gcc11.2' or 'lib/aarch64-ubuntu-gcc9.4'
# We will check common paths or use environment variable SNPE_TARGET_ARCH
snpe_target_arch = os.environ.get('SNPE_TARGET_ARCH', 'aarch64-oe-linux-gcc11.2')
library_dirs = [
    os.path.join(snpe_root, 'lib', snpe_target_arch)
]

libraries = ['SNPE']

ext_modules = [
    Extension(
        'snpe_native',
        ['snpe_yolo_detector.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-fPIC'],
        extra_link_args=['-Wl,-rpath,' + library_dirs[0]]
    ),
]

setup(
    name='snpe_native',
    version='0.1.0',
    description='SNPE PyBind11 wrapper for YOLO inference',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['pybind11>=2.5.0'],
    zip_safe=False,
)
