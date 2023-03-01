import sys
sys.path.extend("./")


from agent.agent import Agent


agent = Agent("192.168.31.138")
agent.run()
