import argparse

"""
Makes a DeepQ Agent and runs it through one fight for each character in the roster so the user can view it
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Imports the specified class library and loads the specificed model.')
    parser.add_argument('-cn', '--className', type= str, default= "DeepQAgent", help= 'Name of the class library to import')
    parser.add_argument('-mn', '--modelName', type= str, default= None, help= 'Name of the specific model to be loaded and tested')
    parser.add_argument('-l', '--load', action= 'store_true', help= 'Boolean flag for if the user wants to load pre-existing weights')
    parser.add_argument('-c', '--character', type= str,  default= "ryu", help= 'The specific character this agent will play')

    args = parser.parse_args()
    if args.className is None:
        args.modelName = args.className

    agent = None
    exec("from {0} import {0}".format(args.className))
    exec("agent = {0}(load= {1}, name= \"{2}\", character= \"{3}\")".format(args.className, args.load, args.modelName, args.character))
    from Lobby import Lobby
    testLobby = Lobby()
    testLobby.addPlayer(agent)
    testLobby.executeTrainingRun(review= False, render= True)
