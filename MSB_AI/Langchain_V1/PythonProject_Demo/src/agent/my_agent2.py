from langchain.agents import create_agent

from agent.my_llm import llm
from agent.toots.tool_demo1 import web_search
from agent.toots.tool_demo2 import MyWebSearchTool

web_search_tool = MyWebSearchTool()  # 创建一个自定义的工具

agent = create_agent(
    llm,
    # tools=[web_search],
    tools=[web_search_tool],
    system_prompt="你是一个智能助手，尽可能的调用工具回答用户的问题。",
)

agent.ainvoke()