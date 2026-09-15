# Source this file to use the local Octopus installation.
export PATH="$HOME/.local/opt/octopus-16.4/bin:$PATH"

# Chemfarm Linux build: use the matching MPI and shared-library runtime.
if [ "$(uname -s)" = Linux ] && [ -d /apps01/apps/openmpi-4.1.5-gcc13 ]; then
    export PATH="/apps01/apps/openmpi-4.1.5-gcc13/bin:$PATH"
    export LD_LIBRARY_PATH="/apps01/apps/gcc/gcc-13.2.0/lib64:/apps01/apps/openmpi-4.1.5-gcc13/lib:/apps01/apps/fftw-3.3.10/lib:/apps01/apps/gsl-2.8/lib:/apps01/apps/libxc-6.2.2/lib64:/apps01/apps/intel/2023/mkl/2023.2.0/lib/intel64:/apps01/apps/intel/2023/compiler/2023.2.1/linux/compiler/lib/intel64_lin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
