The Trello board containing the current tasks and milestones can be found here: https://trello.com/b/bLHQK3YK/street-fighter-ai

# About this Project

## Introduction

This project aims to create a customizable training environment that allows for rapid prototyping, training, and evaluation of AI models learing to play Street Fighter 2 championship edition via competetive tournaments. This training platform is also portable for a variety of retro fighting games with minimal background setup.  

## Milestones
This project has three main goals:  
1. Developing a portable training platform.
2. Developing a competitive tournament matchmaking system.  
3. Developing a suite of analytical tools.

### Portable Platform
This goal aims to design and implement a clear and concise interface between a backend training environment and user built models. A user's custom model needs to inherit from the Agent class which has four abstract functions it's children have to implement to stick to the interface. The user can implement any functionality they want inside their model as long as the interface between the Agent and the training environment is adhered to. The hope is to allow a broad range of models to be quickly developed that can make use of the same training platform without having to replicate work done on setting up the backend.

### Competetive Tournaments
This goal aims to design a system wherein various user models can compete against one another in a simulated tournament to be placed on a leaderboard ranking their performances.  In single player mode a model eventually learns to overfit by learning the state machine behind Street Fighter's AI and can win every stage. So by training against a variety of opponents who are also improving the hope is that it will allow each model a much more realistic and dynamic training environment. Games can even be logged and used as training data for future models who did not even play that match, akin to an athlete watching a highlight real and drawing knowledge from it.

### Analytical Support
This goal aims to create a comprehensive suite of data collection tools that can allow for easy monitoring of a model to evaluate which aspects of the game it is improving on.

## Repo Organization
This section will explain the organization of the repo, if you are trying to install the dependancies then skip to the next section.  

The top level of the repo contains five main directories:

### src
This directory contains all of the source related to the engine, machine learning algorithms, and helper scripts

### tests
This directory contains test code for each code file in src. RUN_TESTS.py can be run that will run each test in the directory, print the results, and then will return whether testing passed or failed. Branch protection will auto run this on any merge and will reject any merge that does not pass the testing.

### local_models
This directory contains a set of unique directories containing model checkpoints for each separate model that is trained locally. These models are not saved online to avoid merge conflicts.

### local_logs
This directory contains a set of unique directories of training logs for each separate model that is trained locally. These logs are not saved online to avoid merge conflicts.

### examples
This directory contains a set of basic examples that demonstrate basic functionality of several libraries used in the source code. New features usually start off as example scripts that serve as launching off points for development. The readme in the directory as a short description of what each example is demonstrating.

### StreetFighterIISpecialChampionEdition-Genesis
This folder contains the ROM, saved game states, environment descriptions, and RAM look ups needed to be fed into the emulator in order for the game engine to run properly.  

And also in the top level of the repo are two files. First is the requirements.txt file. This file contains all of the dependancies needed to run this project. The next section will walk you through installation of those dependancies. 

Next is the software license that defines the legality of using this software for commercial, educational, or private purposes.

After that is the Code of Conduct for working on this project, and finally there is the contribution guidelines to help you get started in contributing to this project if you have ideas for improvements.

---
# Getting Started

This section will take you through how to setup the repo, run an example Agent, make your own Agents, and how to create your own save states to train your Agents on. 

---
## Installing Dependancies

To download the necessary dependencies after cloning move into the top level of the repo and call:  

`pip3 install -r requirements.txt`

These libraries can sometimes have serious issues installing themselves or their dependencies on a windows machine. If you run into trouble that doesn't seem easily fixed it is recommended to work on Linux. As long as you the dependancies get installed on your host machine the code should be cross platform. **Note: this will overwrite the versions of any of these libraries that already exist on your host machine.** If you wish to avoid this it is recommended to use a personal python virtual environment.

---
## Preparing the Game Files 

After the dependencies have been installed the necessary game files, all zipped inside of the **StreetFighterIISpecialChampionEdition-Genesis** directory, can be setup. The game files need to be copied into the data folder of the retro library installation on your local machine. This location can be found by running the following lines in the command line:  

`python3`  
`import retro`  
`print(retro.__file__)`    

That should return the path to where the retro __init__.py script is stored, but this isn't where the game files should be added. One level up from that should be the data folder. Inside there should be the stable folder. Copy the **StreetFighterIISpecialChampionEdition-Genesis** folder that is in the top level of the repo here. Inside the folder should be the following files:

1. rom.md    
2. rom.sha    
3. scenario.json  
4. data.json  
5. metadata.json  
6. reward_script.lua   
7. Several .state files split into two different categories:  
    1. single_player states that load up one Agent into a stage from the actual single player mode of the game for the specified character  
    2. two_player states that load up two Agents to play against one another  

With that the game files should be correctly set up and you should be able to run a test agent. 

---
## The first test run

To double check that the game files were properly set up the example agent can be run. cd into the src directory. Then either run the following command on your terminal or from your preferred IDE:

`python3 Agent.py -r`

The -r sets the render flag so we can visualize it working. Without that flag the Agent will play the games but they will not render. This Agent should essentially just button mash and look like it is playing randomly. If everything was installed correctly it should run through each level of the single player game as ryu. A small window will pop up showing the current stage. Once the fight is over a new window should open up with the next stage. Once all stages are over the program should kill itself and close all windows.

---
## How to make an agent

To make your own agent it is required that your model inherits from Agent.py and adheres to the specific interface defined within. The goal is to create a streamlined platform to rapidly prototype, train, and deploy new agents instead of starting for scratch every time. As well enforcing the interface for the agent class allows for high level software to be developed that can import various user created agents without fear of breaking due to specific interface issues. 

### Input Parameters of the Agent Interface

There are four parameters your Agent has to be able to accpet when it's constructor is called.

1. load - a boolean that specifices whether or not to load a pretrained model or start from scratch with this agent
2. name - a string representing the name of the model, will be used when creating directories for it's training checkpoints
3. character - a string representing which character in the fighting game roster this agent will play as
4. verbose - a boolean that specifies whether or not to print various debug statements to the console during execution

These four parameters must be passed into the super constructor of the Agent class.

### Abstract Methods of the Agent Interface

There are four main functions that need to be implemented in order to create a new user agent.

1. getMove  
2. initializeNetwork  
3. prepareMemoryForTraining  
4. trainNetwork

Each section below gives a description of the interface required for each function and it's purpose. Further documentation can be seen inside the code of Agent.py

#### getMove

Get move takes in the current image oberservation of the game and the RAM data and must return an integer that represents the index into the discrete action space for the button combination the agent wants to press. A full list of the discrete action space can be found inside Discretizer.py

#### initializeNetwork

initializeNetwork creats and initializes the model's underlying network and returns it, if the load flag is supplied when initializing an Agent then a previously pretrained model will be loaded instead and this function does not need to be implemented. It does not take in any parameters but its expected to return the model desired for the Agent.

#### prepareMemoryForTraining

As the Agent plays it records the events during a fight. It records observation, state, action, reward, next observation, next state reward sequences. Each index in the memory buffer of the Agent demonstrates a state the Agent was presented with, the action it took, the next state the action led to, the reward the Agent received for that action, and a flag specifying if that game instance is finished. State and next state are both dictionaries containing the RAM data of the game at those times as specified in Data.json. The action is an integer representing the index in the action space of the button combination the Agent chose to execute that frame. And finally Done is a boolean flag where True means the current game instance is over. Is expected to return an array containing the full set of prepared training data. There is no specific format this data must be in when returned, it is immediately passed to your trainNetwork function right after. The elements of each data point fed to this function to prepare may change over time but their indices are stored in a set of static variables in Agent.py that you can use:

-OBSERVATION_INDEX   
-STATE_INDEX   
-ACTION_INDEX   
-REWARD_INDEX   
-NEXT_OBSERVATION_INDEX   
-NEXT_STATE_INDEX   
-DONE_INDEX   

A child class can access them by calling {class_name}.{variable_name}, where {class.name} is the name of your child class. Indexing into a step to get the next observation from the DeepQAgent for example would look like:

`step[DeepQAgent.NEXT_OBSERVATION_INDEX]`

#### trainNetwork

Takes in the prepared training data and the current model and runs the desired amount of training epochs on it. The trained model must then ;be returned once training is finished.

---

### Training Checkpoints

Once a round of training is complete and the updated model is returned a checkpoint will be made by Agent.py that saves the trained model as a backup. As well a custom training log will be that will show the training error of the Agent as it is learning. These logs and models are stored in their own unique logs and model directories based on the name of their model. The naming convention is local_models/{class_name}/{class_name}.model and local_models/{class_name}/{class_name}.log. A model can be given a name upon initialization, if none is given it's class.name variable will default to the class name itself. The local models folder is used to only train models locally and the git ignore inside prevents these models from being tracked so to avoid merge conflicts. There is a pretrained models folder that example test models can be put inside.

### Watch Agent

Watch Agent is a basic script that allows the user to qucikly load in a specific Agent and visualize it playing through the single player mode without having to setup an entire tournament. It is useful to use this in conjunction with checkpoints in order to pause an Agent between episodes and view it's progress to understand if it's headed in the right direction and if things seem to be working correctly.

## Json Files

There are three json files and a lua script that the gym environment reads in order to setup the emulation environment. These files are metadata.json, data.json, scenario.json, and reward_script.lua. 

### Metadata.json

The metadata.json file can hold high level global information about the game environment. For now this simply tells the environment the default save state that the game ROM should launch in if none has been selected. 

### Data.json

The data.json file is an abstraction of the games ram into callable variables with specified data types that the environment, user, and environment.json files can interact with. A complete list of named variables and their corresponding addresses in memory can be found listed in the file itself. If a publicly available RAM dump for a game can not be found finding new variables on your own is an involved process and requires monitoring RAM and downloading the bizhawk emulator. Bizhawk is an emulator used for developing tool assisted speedruns and has a wide selection of tools for RAM snooping. This video is a good reference for learning how to snoop RAM:

https://www.youtube.com/watch?v=zsPLCIAJE5o&t=900s

### Scenario.json

Scenario.json specifies several hyper parameters describing the game environment, such as when it is finished, which reward script is being used, and how large the emulator window will be. The reward function for the StreetFighterAgents is seperated into it's own lua script to make designing a more complex reward function easier. The script is imported for use by gym-retro's environment via the code snippet in scenario.json as follows:

```

"reward": {
        "script": "lua:calculate_reward"
    },
"scripts": [
        "reward_script.lua"
    ],

```

#### Reward Function

The reward function specifies what game state variables factor into the reward and what weights are assigned to them. Each game state variable can either positively or negatively affect the reward for each player. After each action is taken by both players a reward is calculated and returned to both players. The reward is mirrored for both players so if player 1 recieves a reward of positive 100 then player 2 will receive a score of -100 because they are in direct competition with one another. The reward for that time step is then recorded and stored for later training after all fights in an epoch are finished. For now the default reward function utilizes game state variables such as the change in each player's health and whether a round was won or lost on that time step. 

#### Done

Done is a flag that signifies whether the current environment has completed. If True the match will be ended the Agents will then be given time to train. Currently Done is set to True once either player has won two rounds. 

---
## Generating New Save States

Save states are generated by a user actually saving the current game state while running the rom in an emulator In order to make new save states to contribute to the variety of matches your Agent will play in you have to actually play the Street Fighter ROM up until the point you want the Agent to start at. 

### Installing the Emulator

Retroarch is the emulator that is needed to generate the correct save states under the hood. It can be installed at:  
https://www.retroarch.com/?page=platforms


### Preparing the Cores

Retroarch needs a core of the architecture it is trying to simulate. The Street Fighter ROM we are working with is for the Sega Genisis. Retro actually has a built in core that can be copy and pasted into Retroarchs core folder and this is their recommended installation method. However finding the retroarch installation folder can be difficult and so can finding the cores in the Retro library. Instead open up Retroarch and go into Load Core. Inside Load Core scroll down and select download core. Scroll way down until you see genesis_plus_gx_libretro.so.zip and install it. Now go back to the main menu and select Load Content. Navigate to the Street Fighter folder at the top level of the repo and load the rom.md file. From here the game should load up correctly.

### Saving states

F2 is the shortcut key that saves the current state of the game. The state is saved to the currently selected game state slot. This starts at slot zero and can be incremented with the F6 key and decremented with the F7 key. When a fight is about to start that you want to create a state for hit F2. Then it is recommended to increment the save slot by pressing F6 so that if you try to save another state you don't accidentally overwrite the last state you saved. There are 8 slots in total. By pressing F5 and going to view->settings-Directory you can control where the save states are stored. The states will be saved with the extension of 'state' plus the number of the save slot it was saved in. To prep these for usage cleave off the number at the end of each extension and rename each file in accordance to the single player and two player formats that can be seen with existing states. Then move these ROMS into the game files inside of retro exactly like when preparing the game files after the initial cloning of the repo. Once inside that repo each state should be zipped independently of one another. Once this happens the extension will now be .zip, remove this from the extension so that the extension still remains .state. The states are now ready to be loaded by the agent. 

---
## Example Implemented Agents

There are two examples of already implemented Agents. 

### DeepQAgent

First is DeepQAgent. DeepQAgent is an implementation of the DeepQ reinforcement algorithm for playing StreetFighter using policy gradients, a dense reward function, and greedy exploration controlled by an epsilon value that decreases as the model is trained. Each action the Agent takes during a fight is rewarded after a change in time in order to see what effect the move had on the fight outcome. When the model is first initialized it plays completely randomly in order to kick start rapid greedy exploration. As the model trains epsilon slowly decreases until the model begins to take over now that it has watched random play for a while and hopefully picked up some techniques. Below is a description of the implementation of each of the abstract functions required for a child class. The observation of the current state is not actually used in training as building a network to do feature extraction for each stage and fighter combination from image data would be incredibly hard. That information is thrown away and instead only the RAM data is used to train.

#### Get Move

The RAM data of the current state is converted into a feature vector by the helper function prepareNetworkInputs. Several parts of the RAM data such as the enemy character, player and enemy_state, and more are one hot encoded so that mutual independence is established. This leads to a feature vector containing 30 elements that is fed into the network. The output of the network decides what move the Agent will take from a preset move list where the move is then mapped to a set of inputs via a look up table and fed into the environment. The activation function of the output layer is a softmax function that assigns a probability to each possible move. These probabilities all sum to one and the move with the highest probability is picked as the action the Agent will undertake. However whenever a move is requested by the Agent a random number is generated, if this number is below the epsilon value, which ranges from 0 to 1 and decreases over time towards a lower bound during training, a random move is picked instead. This forces exploration of new strategies by the Agent. However this exploration is not informed by any model the Agent has and so is simply random greedy exploration. 

#### initializeNetwork

The input layer is the size of the info about the state space that is created in prepareNetworkInputs when converting the RAM info of the current state into a feature vector. There are several hidden layers all with linear activations. The output layer is the size of the predefined user action space. Note that this is not the same as the action space of the game. The action space of the game is the number of buttons the Agent can possibly press, this action space is far too large to explore and so a set of "combos" has been made inside the discretizer that correlates to each meaningful set of button presses in the game. So the output of the network is the size of this predefined set of button combinations. The output activation function is linear and represents the predicted reward of doing each combo, the move with the highest reward is chosen. 

#### preprareMemoryForTraining

Each time step from the training game memory has the RAM data from the game converted into a 30 element feature vector described in the top of the section. That is the only preparation needed. The observation of the current image data is thrown out as this model only operates on the RAM data.

#### trainNetwork

For training the method of policy gradients is used. A dense reward function has been designed so that the Agent can be given frequent rewards for using good moves. Policy gradients essentially uses the reward for each state, action, new state sequence as the gradient for training our network. The gradient vector then has the index of the button combination chosen set to the observed reward and all other indices are left as predicted.

### HumanAgent

An Agent that actually is controlled by a human. When the Agent is called into a game it turns on a hook into the user's keyboard and enters it's key presses into the game. When the game finishes the hook is turned off. User input is continously sampled and the current pressed keys are updated in an input buffer at any new key event. When the game lobby asks for this Agent's move the last updated input buffer is submitted. 

---
## Further Help

Feel free to open up issues if running into problems with this project. However before asking any issues please check to see if your issue has already been answered. If you have any questions regarding contributing please refer to the contributing guidelines for more information. 

---
## References:
https://github.com/openai/retro/issues/33 (outdated but helpful)   
https://medium.com/aureliantactics/integrating-new-games-into-retro-gym-12b237d3ed75 (Very helpful for writing the json files) 
https://github.com/keon/deep-q-learning (Someones basic implementation of DeepQ in python) 

https://www.youtube.com/watch?v=JgvyzIkgxF0 (reinforcement learning intro video)   
https://www.youtube.com/watch?v=0Ey02HT_1Ho (more advanced techniques)   
http://karpathy.github.io/2016/05/31/rl/ (good article on basic reinforcement learning)   
https://towardsdatascience.com/reinforcement-learning-lets-teach-a-taxi-cab-how-to-drive-4fd1a0d00529 (article on deep q learning for learning atari games)   
