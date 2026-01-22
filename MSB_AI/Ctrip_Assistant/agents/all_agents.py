from langchain_core.tools import InjectedToolCallId
from langgraph.graph import MessagesState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from tools.init_db import update_db
from llm_libs import llm
from langchain.agents import create_agent
from pydantic import BaseModel,Field
from tools.search_tool import MySearchTool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from typing import Annotated

# 这是整个状态图的完整内存
memory = InMemorySaver()

# 搜索 agent
research_agent = create_agent(
    model=llm,
    tools=[MySearchTool()],
    system_prompt=(
        "你是一个网络搜索的智能体(Agent)。\n\n"
        "指令：\n"
        "- 仅网络数据获取、网络查询、数据查询相关的任务\n"
        "- 回复时仅包含工作结果，不要包含任何其他文字"
    ),
    name="research_agent",
)

# Wrap it as a tool
@tool("research", description="Research a topic and return findings")
def call_research_agent(state: Annotated[MessagesState, InjectedState],tool_call_id: Annotated[str, InjectedToolCallId],)-> Command:
    agent_name = "research_agent"
    name = f"transfer_to_{agent_name}"
    """
    执行实际的转接操作。

    创建一个工具消息表明转接成功，并返回一个命令对象指示流程控制器
    将控制权转移给指定代理，同时更新会话状态。

    参数:
        state (MessagesState): 当前会话状态，包含消息历史等信息
        tool_call_id (str): 工具调用的唯一标识符

    返回:
        Command: 包含转接指令和状态更新的命令对象
    """
    # 构造工具消息，记录转接操作的成功执行
    tool_message = {
        "role": "tool",
        "content": f"Successfully transferred to {agent_name}",
        "name": name,
        "tool_call_id": tool_call_id,
    }
    # result = research_agent.invoke({"messages": [{"role": "user", "content": state}]})
    #
    return Command(
            goto="research_agent",
            update={**state, "messages": state["messages"] + [tool_message] },
            graph=Command.PARENT,
        )

# 主 agent
supervisor_agent = create_agent(
    model=llm,
    tools=[call_research_agent],
    system_prompt=(
        "你被调用时，请直接调用 call_research_agent 工具"
        # "你是一个监督者或者管理者，管理一个智能体：\n"
        # "- 网络搜索智能体：分配与网络搜索、数据查询相关的任务\n"
        # "- 航班预订能体：分配与航班查询，预定，改签等相关的任务\n"
        # "- 酒店预订智能体：分配与酒店查询，预定，修改订单等相关的任务\n"
        # "- 汽车租赁预定智能体：分配与汽车租赁查询，预定，修改订单等相关的任务\n"
        # "- 旅行产品预定智能体：分配与旅行推荐查询，预定，修改订单等相关的任务\n"
        # "处理规则：\n"
        # "1. 如果问题属于以下类别，直接回答：\n"
        # "   - 可以根据上下文记录直接回答的内容（如'你的航班信息，起飞时间等'）。\n"
        # "   - 不需要工具的一般咨询（如'你好'）。\n"
        # "   - 确认类问题（如'你收到我的请求了吗'）。\n"
        # "2. 其他情况按类型分配给对应智能体。\n"
        # "3. 一次只分配一个任务给一个智能体。\n"
        # "4. 不要自己执行需要工具的任务。\n"
    ),
    checkpointer=memory,
    name="supervisor",
)