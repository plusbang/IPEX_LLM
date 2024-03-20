export MASTER_ADDR=127.0.0.1
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

NUM_GPUS=2 # number of used GPU
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0 # Different from PVC

mpirun -np $NUM_GPUS --prepend-rank python run.py
# export CCL_DG2_ALLREDUCE=1 # For internal CCL

# CCL_ROOT=/home/arda/binbin/perf-optimize/oneccl-from-jianwei/rollback-stable-version/1ccl_dg2_allreduce_20240308  LD_LIBRARY_PATH=/home/arda/binbin/perf-optimize/oneccl-from-jianwei/rollback-stable-version/1ccl_dg2_allreduce_20240308/src:/home/arda/binbin/perf-optimize/oneccl-from-jianwei/rollback-stable-version/1ccl_dg2_allreduce_20240308/deps/mpi/lib:/opt/intel/oneapi/2024.0/lib \
#     mpirun -np $NUM_GPUS --prepend-rank python run.py
