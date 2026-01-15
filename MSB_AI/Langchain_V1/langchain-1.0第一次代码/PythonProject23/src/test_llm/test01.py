import time

from agent.my_llm import llm

resp = llm.invoke('用三句话简单介绍一下：机器学习的基本概念')
print(type(resp))
print(resp)
print(time.time())
resp = llm.invoke('用三句话简单介绍一下：深度学习的基本概念')
print(resp)
print(time.time())