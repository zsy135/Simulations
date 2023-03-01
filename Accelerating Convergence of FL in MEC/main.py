def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class A:
    def __init__(self):
        self.a = 1

    def f(self, b):
        print(self.a + b)

    def test(self):
        f = getattr(self, "f")
        return f


from agent.agent import Agent

d = {1:"123","a":234}

Agent.format_write_dict("./test.py", d)

