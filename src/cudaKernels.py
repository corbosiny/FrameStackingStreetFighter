import numpy
from numba import cuda, jit

@cuda.jit
def prepareMemoryForTrainingCuda(memory, action, reward, done):
    thread_id = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    action[thread_id] = memory[0][thread_id]
    reward[thread_id] = memory[1][thread_id]
    done[thread_id] = memory[2][thread_id]