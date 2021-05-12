import argparse, retro, threading, os, numpy, random, math
from Agent import Agent
from LossHistory import LossHistory
from cudaKernels import prepareMemoryForTrainingCuda
from numba import cuda

import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow.keras.losses

from collections import deque

from time import perf_counter

class DeepQAgent(Agent):
    """An agent that implements the Deep Q Neural Network Reinforcement Algorithm to learn street fighter 2"""
    
    EPSILON_MIN = 0.1                                         # Minimum exploration rate for a trained model
    DEFAULT_EPSILON_DECAY = 0.999                             # How fast the exploration rate falls as training persists
    DEFAULT_DISCOUNT_RATE = 0.98                              # How much future rewards influence the current decision of the model
    DEFAULT_LEARNING_RATE = 0.0001

    # Mapping between player state values and their one hot encoding index
    stateIndices = {512 : 0, 514 : 1, 516 : 2, 518 : 3, 520 : 4, 522 : 5, 524 : 6, 526 : 7, 532 : 8} 
    doneKeys = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]

    ACTION_BUTTONS = ['X', 'Y', 'Z', 'A', 'B', 'C']

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        """Implementation of huber loss to use as the loss function for the model"""
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def __init__(self, stateSize= 32, actionSize= 51, load= False, epsilon= 1, name= None, character= "ryu", verbose= True):
        """Initializes the agent and the underlying neural network

        Parameters
        ----------
        stateSize
            The number of features that will be fed into the Agent's network

        actionSize
            The size of the action space the Agent can chose actions in

        load
            A boolean flag that specifies whether to initialize the model from scratch or load in a pretrained model

        epsilon
            The initial exploration value to assume when the model is initialized. If a model is lodaed this is set
            to the minimum value

        name
            A string representing the name of the agent that will be used when saving the model and training logs
            Defaults to the class name if none is provided

        character
            String representing the name of the character this Agent plays as

        verbose
            A boolean variable representing whether or not the print statements in the class are turned on
            Error messages however are not turned off

        Returns
        -------
        None
        """
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE         # discount rate
        if load: self.epsilon = DeepQAgent.EPSILON_MIN        # If the model is already trained lower the exploration rate
        else: self.epsilon = epsilon                          # If the model is not trained set a high initial exploration rate
        self.epsilonDecay = DeepQAgent.DEFAULT_EPSILON_DECAY  # How fast the exploration rate falls as training persists
        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE 
        self.lossHistory = LossHistory()
        super(DeepQAgent, self).__init__(load= load, name= name, character= character, verbose= verbose) 

    def getMove(self, obs, info):
        """Returns a set of button inputs generated by the Agent's network after looking at the current observation

        Parameters
        ----------
        obs
            The observation of the current environment, 2D numpy array of pixel values

        info
            An array of information about the current environment, like player health, enemy health, matches won, and matches lost, etc.
            A full list of info can be found in data.json

        Returns
        -------
        move
            An integer representing the move selected from the move list
        """        
        if numpy.random.rand() <= self.epsilon:
            move = self.getRandomMove()
            return move
        else:
            stateData = self.prepareNetworkInputs(info)
            predictedRewards = self.model.predict(stateData)[0]
            move = numpy.argmax(predictedRewards)
            return move

    def initializeNetwork(self):
        """Initializes a Neural Net for a Deep-Q learning Model
        
        Parameters   
        ----------
        None

        Returns
        -------
        model
            The initialized neural network model that Agent will interface with to generate game moves
        """
        model = Sequential()
        model.add(Dense(48, input_dim= self.stateSize, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(192, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.actionSize, activation='linear'))
        model.compile(loss=DeepQAgent._huber_loss, optimizer=Adam(lr=self.learningRate))

        if self.verbose: print('Successfully initialized model')
        return model

    def prepareMemoryForTraining(self, memory):
        """prepares the recorded fight sequences into training data
        
        Parameters
        ----------
        memory
            A 2D array where each index is a recording of a state, action, new state, and reward sequence
            See readme for more details

        Returns
        -------
        data
            The prepared training data in whatever from the model needs to train
            DeepQ needs a state, action, and reward sequence to train on
            The observation data is thrown out for this model for training
        """
        
        startTimer = perf_counter()

        data = []
        for step in self.memory:
            data.append(
            [self.prepareNetworkInputs(step[Agent.STATE_INDEX]), 
            step[Agent.ACTION_INDEX], 
            step[Agent.REWARD_INDEX],
            step[Agent.DONE_INDEX],
            self.prepareNetworkInputs(step[Agent.NEXT_STATE_INDEX])])
        
        print("Elapsed time serial: " + str(perf_counter() - startTimer) + '\n')
 
        # Pre-processing memory array here to overcome CUDA Python limitations
        cudaMemory = numpy.array([[row[Agent.ACTION_INDEX] for row in self.memory],                             # 0
                                 [row[Agent.REWARD_INDEX] for row in self.memory],                              # 1
                                 [row[Agent.DONE_INDEX] for row in self.memory]])                               # 2

        cudaStateMemory = numpy.array([[row[Agent.STATE_INDEX]["player1_health"] for row in self.memory],   # 0
                                 [row[Agent.STATE_INDEX]["player1_x_position"] for row in self.memory],     # 1   
                                 [row[Agent.STATE_INDEX]["player1_y_position"] for row in self.memory],     # 2
                                 [row[Agent.STATE_INDEX]["player1_character"] for row in self.memory],      # 3   
                                 [row[Agent.STATE_INDEX]["player1_status"] for row in self.memory],         # 4   
                                 [row[Agent.STATE_INDEX]["player2_health"] for row in self.memory],         # 5
                                 [row[Agent.STATE_INDEX]["player2_x_position"] for row in self.memory],     # 6
                                 [row[Agent.STATE_INDEX]["player2_y_position"] for row in self.memory],     # 7
                                 [row[Agent.STATE_INDEX]["player2_character"] for row in self.memory],      # 8
                                 [row[Agent.STATE_INDEX]["player2_status"] for row in self.memory]])        # 9
                                                                
        cudaNextStateMemory = numpy.array([[row[Agent.NEXT_STATE_INDEX]["player1_health"] for row in self.memory],
                                 [row[Agent.NEXT_STATE_INDEX]["player1_x_position"] for row in self.memory],
                                 [row[Agent.NEXT_STATE_INDEX]["player1_y_position"] for row in self.memory],
                                 [row[Agent.NEXT_STATE_INDEX]["player1_character"] for row in self.memory], 
                                 [row[Agent.NEXT_STATE_INDEX]["player1_status"] for row in self.memory],                  
                                 [row[Agent.NEXT_STATE_INDEX]["player2_health"] for row in self.memory],
                                 [row[Agent.NEXT_STATE_INDEX]["player2_x_position"] for row in self.memory],
                                 [row[Agent.NEXT_STATE_INDEX]["player2_y_position"] for row in self.memory],
                                 [row[Agent.NEXT_STATE_INDEX]["player2_y_position"] for row in self.memory],
                                 [row[Agent.NEXT_STATE_INDEX]["player2_character"] for row in self.memory],
                                 [row[Agent.NEXT_STATE_INDEX]["player2_status"] for row in self.memory]])

        actionArr = numpy.zeros(len(self.memory))
        rewardArr = numpy.zeros(len(self.memory))
        doneArr = numpy.zeros(len(self.memory), dtype=bool)
        stateArr = numpy.zeros((len(self.memory), 3 + (2*len(DeepQAgent.stateIndices.keys())) + 8 + 3))
        nextStateArr = numpy.zeros((len(self.memory), 3 + (2*len(DeepQAgent.stateIndices.keys())) + 8 + 3))

        playerNum = numpy.array([self.playerNumber])
        doneKeys = numpy.array(DeepQAgent.doneKeys)
        stateIndices = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8])

        # Copy arrays from host to device memory (blocking calls)
        startTimer = perf_counter()
        d_cudaMemory = cuda.to_device(cudaMemory)
        d_stateMemory = cuda.to_device(cudaStateMemory)
        d_nextStateMemory = cuda.to_device(cudaNextStateMemory)
        d_actionArr = cuda.to_device(actionArr)
        d_rewardArr = cuda.to_device(rewardArr)
        d_doneArr = cuda.to_device(doneArr)
        d_stateArr = cuda.to_device(stateArr)
        d_nextStateArr = cuda.to_device(nextStateArr)
        d_playerNum = cuda.to_device(playerNum)
        d_doneKeys = cuda.to_device(doneKeys)
        d_stateIndices = cuda.to_device(stateIndices)
        hostToDeviceTime = perf_counter() - startTimer
        print("Elapsed time host-to-device: " + str(hostToDeviceTime) + '\n')        

        # Number of threads per block
        threadsPerBlock = 32

        # Number of blocks per grid
        blocksPerGrid = len(cudaMemory) + (threadsPerBlock - 1)

        startTimer = perf_counter()
        # Invoke the CUDA kernel (blocking call)
        prepareMemoryForTrainingCuda[blocksPerGrid, threadsPerBlock](d_cudaMemory, d_stateMemory, d_nextStateMemory, d_actionArr, d_rewardArr, d_doneArr, d_stateArr, d_nextStateArr, d_playerNum, d_doneKeys, d_stateIndices)

        print("Elapsed time parallel: " + str(perf_counter() - startTimer - hostToDeviceTime) + '\n')

        # Arrays are automatically copied back from device to host
        # when kernel finishes so we'll use hostToDeviceTime as an estimate        

        print(d_stateArr[0][0])
        print(d_actionArr[0])
        print(d_rewardArr[0])
        print(d_doneArr[0])
        print(d_nextStateArr[0][0])
        print(data[0])
        return data

    def prepareNetworkInputs(self, step):
        """Generates a feature vector from the current game state information to feed into the network
        
        Parameters
        ----------
        step
            A given set of state information from the environment
            
        Returns
        -------
        feature vector
            An array extracted from the step that is the same size as the network input layer
            Takes the form of a 1 x 30 array. With the elements:
            enemy_health, enemy_x, enemy_y, 8 one hot encoded enemy state elements, 
            8 one hot encoded enemy character elements, player_health, player_x, player_y, and finally
            8 one hot encoded player state elements.
        """
        feature_vector = []
        
        # Enemy Data
        if self.playerNumber == 0: enemyKey = 2
        else: enemyKey = 1
        feature_vector.append(step["player{0}_health".format(enemyKey)])
        feature_vector.append(step["player{0}_x_position".format(enemyKey)])
        feature_vector.append(step["player{0}_y_position".format(enemyKey)])

        # one hot encode enemy state
        # enemy_status - 512 if standing, 514 if crouching, 516 if jumping, 518 blocking, 522 if normal attack, 524 if special attack, 526 if hit stun or dizzy, 532 if thrown
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        statusKey = 'player{0}_status'.format(enemyKey)
        if step[statusKey] not in DeepQAgent.doneKeys: oneHotEnemyState[DeepQAgent.stateIndices[step[statusKey]]] = 1
        feature_vector += oneHotEnemyState

        # one hot encode enemy character
        oneHotEnemyChar = [0] * 8
        oneHotEnemyChar[step["player{0}_character".format(enemyKey)]] = 1
        feature_vector += oneHotEnemyChar

        # Player Data
        feature_vector.append(step["player{0}_health".format(self.playerNumber + 1)])
        feature_vector.append(step["player{0}_x_position".format(self.playerNumber + 1)])
        feature_vector.append(step["player{0}_y_position".format(self.playerNumber + 1)])

        # player_status - 512 if standing, 514 if crouching, 516 if jumping, 520 blocking, 522 if normal attack, 524 if special attack, 526 if hit stun or dizzy, 532 if thrown
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        statusKey = 'player{0}_status'.format(self.playerNumber + 1)
        if step[statusKey] not in DeepQAgent.doneKeys: oneHotPlayerState[DeepQAgent.stateIndices[step[statusKey]]] = 1
        feature_vector += oneHotPlayerState

        feature_vector = numpy.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def trainNetwork(self, data, model):
        """To be implemented in child class, Runs through a training epoch reviewing the training data
        Parameters
        ----------
        data
            The training data for the model to train on, a 2D array of state, action, reward, sequence

        model
            The model to train and return the Agent to continue playing with
        Returns
        -------
        model
            The input model now updated after this round of training on data
        """
        minibatch = random.sample(data, len(data))
        self.lossHistory.losses_clear()
        for state, action, reward, done, next_state in minibatch:     
            modelOutput = model.predict(state)[0]
            if not done:
                reward = (reward + self.gamma * numpy.amax(model.predict(next_state)[0]))

            modelOutput[action] = reward
            modelOutput = numpy.reshape(modelOutput, [1, self.actionSize])
            model.fit(state, modelOutput, epochs= 1, verbose= 0, callbacks= [self.lossHistory])

        if self.epsilon > DeepQAgent.EPSILON_MIN: self.epsilon *= self.epsilonDecay
        return model

from keras.utils.generic_utils import get_custom_objects
loss = DeepQAgent._huber_loss
get_custom_objects().update({"_huber_loss": loss})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Processes agent parameters.')
    parser.add_argument('-r', '--render', action= 'store_true', help= 'Boolean flag for if the user wants the game environment to render during play')
    parser.add_argument('-l', '--load', action= 'store_true', help= 'Boolean flag for if the user wants to load pre-existing weights')
    parser.add_argument('-e', '--episodes', type= int, default= 10, help= 'Intger representing the number of training rounds to go through, checkpoints are made at the end of each episode')
    parser.add_argument('-n', '--name', type= str, default= None, help= 'Name of the instance that will be used when saving the model or it\'s training logs')
    args = parser.parse_args()
    qAgent = DeepQAgent(load= args.load, name= args.name)

    from Lobby import Lobby
    testLobby = Lobby()
    testLobby.addPlayer(qAgent)
    testLobby.executeTrainingRun(episodes= args.episodes, render= args.render)
