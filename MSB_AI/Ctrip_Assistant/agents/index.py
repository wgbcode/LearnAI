from langchain_core.messages import HumanMessage
from tools.init_db import update_db
from draw_png import draw_graph
from langgraph.graph import StateGraph,MessagesState
from all_agents import  supervisor_agent,research_agent
from langgraph.constants import END, START
from langgraph.types import Command
from typing_extensions import  Annotated
import operator
from typing import List
from typing_extensions import TypedDict

# 只在初始化时执行一次
# update_db()

# 创建工作流
# Graph state
class State(MessagesState):
    pass

workflow_chain = (StateGraph(State)
            .add_node("supervisor", supervisor_agent,destinations=("research_agent",END))
            .add_node("research_agent", research_agent,destinations=(END,))
            .add_edge(START, "supervisor")
            .compile())

# 画出工作流
# draw_graph(workflow_chain, "workflow.png")

# 必须传 Command 对象，要不后面的 agent 会拿不到 message
resp = workflow_chain.invoke(Command(update={"messages": [HumanMessage(content="今天南宁天气怎样？")]}))
print(resp)


