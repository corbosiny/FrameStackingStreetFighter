from DeepQAgent import DeepQAgent
import argparse

"""
Makes a DeepQ Agent and runs it through one fight for each character in the roster so the user can view it
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Imports the specified class library and loads the specificed model.')
    parser.add_argument('className', type= str, default= DeepQAgent, help= 'Name of the class library to import')
    parser.add_argument('-mn', '--modelName', type= str, default= None, help= 'Name of the specific model to be loaded and tested')

    args = parser.parse_args()
    if args.modelName is None:
        args.modelName = args.className

    agent = None
    eval("from {0} import {0}".format(args.className))
    eval("agent = {0}(load= True, name= {1})".format(args.className, args.modelName))

    from Lobby import Lobby
    testLobby = Lobby()
    testLobby.addPlayer(agent)
    testLobby.executeTrainingRun(review= False, render= True)
