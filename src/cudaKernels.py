import numpy
from numba import cuda, jit

@cuda.jit
def prepareMemoryForTrainingCuda(memory, memState, memNextState, action, reward, done, state, nextState, playerNum, doneKeys, stateIndices):
    thread_id = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    
    # Action, reward, and done status
    action[thread_id] = memory[0][thread_id]
    reward[thread_id] = memory[1][thread_id]
    done[thread_id] = memory[2][thread_id]
    
    # Health, x position, y position
    if playerNum[0] == 0: startIndex = 5
    else: startIndex = 0
    nextIndex = 0
    state[thread_id][0] = memState[startIndex][thread_id]
    state[thread_id][1] = memState[startIndex + 1][thread_id]
    state[thread_id][2] = memState[startIndex + 2][thread_id]

    nextState[thread_id][0] = memNextState[startIndex][thread_id]
    nextState[thread_id][1] = memNextState[startIndex + 1][thread_id]
    nextState[thread_id][2] = memNextState[startIndex + 2][thread_id]

    nextIndex += 3
    
    # Enemy status
    if playerNum[0] == 0: statusIndex = 9
    else: statusIndex = 4
    statusKey = memState[statusIndex][thread_id]
    found = False
    for i in doneKeys:
        if statusKey == i:
            found = True
            break
    if not found:
        state[thread_id][stateIndices[statusKey] + nextIndex] = 1

    
    statusKey = memNextState[statusIndex][thread_id]
    found = False
    for i in doneKeys:
        if statusKey == i:
            found = True
            break

    if not found:
        nextState[thread_id][stateIndices[statusKey] + nextIndex] = 1
    
    nextIndex += 8
    
    # Enemy character
    state[memState[startIndex + 3][thread_id] + nextIndex] = 1

    nextState[memNextState[startIndex + 3][thread_id] + nextIndex] = 1

    nextIndex += 1

    # Player data
    if playerNum[0] == 0: startIndex = 0
    else: startIndex = 5
    
    state[thread_id][nextIndex] = memState[startIndex][thread_id]
    state[thread_id][nextIndex + 1] = memState[startIndex + 1][thread_id]
    state[thread_id][nextIndex + 2] = memState[startIndex + 2][thread_id]

    nextState[thread_id][nextIndex] = memNextState[startIndex][thread_id]
    nextState[thread_id][nextIndex + 1] = memNextState[startIndex + 1][thread_id]
    nextState[thread_id][nextIndex + 2] = memNextState[startIndex + 2][thread_id]
    
    nextIndex += 3
    
    # Player status
    if playerNum[0] == 0: statusIndex = 4
    else: statusIndex = 9
    statusKey = memState[statusIndex][thread_id]
    found = False
    for i in doneKeys:
        if statusKey == i:
            found = True
            break
    if not found:
        state[thread_id][stateIndices[statusKey] + nextIndex] = 1

    statusKey = memNextState[statusIndex][thread_id]
    found = False
    for i in doneKeys:
        if statusKey == i:
            found = True
            break
    
    if not found:
        nextState[thread_id][stateIndices[statusKey] + nextIndex] = 1
    
