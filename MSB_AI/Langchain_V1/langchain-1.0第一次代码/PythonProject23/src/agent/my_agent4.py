from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agent.my_llm import llm

from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):
    topic: str  # 存了报幕词的主题

@tool
def update_topic_info(
    runtime: ToolRuntime,
) -> Command:
    """可以查到信息"""
    print(f'config: {runtime.config}')
    topic_type = runtime.config.get('configurable').get('topic_type', None)
    name = "相声" if topic_type == "1" else "小品"
    return Command(update={
        "topic": name,
        # 更新消息历史
        "messages": [
            ToolMessage(
                "成功查找到报幕词的主题",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def generate(
    runtime: ToolRuntime
) -> str | Command:
    """这是一个专门生成报幕词的工具。"""
    print(f'运行时的状态: {runtime.state}')
    print(f'当前工具的调用ID: {runtime.tool_call_id}')
    topic = runtime.state.get("topic", None)
    if topic is None:
       return Command(update={  # command
            "messages": [
                ToolMessage(
                    "请调用'update_topic_info'工具，它将获取并更新报幕词主题。",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"大家好，接下来请欣赏著名艺术家的作品：{topic}!"

agent = create_agent(
    llm,
    tools=[update_topic_info, generate],
    state_schema=CustomState,
    system_prompt="""
    你是一个专门生成报幕词的AI智能体。
    请始终使用 generate 获取报幕词。
    """,
)

if __name__ == '__main__':
    # system_msg = SystemMessage("You are a helpful coding assistant.")
    # messages = [
    #     system_msg,
    #     HumanMessage("How do I create a REST API?")
    # ]
    for step in agent.stream(
            input={'messages': [{'role': 'user', 'content': '帮我生成一个报幕词。'}]},
            stream_mode="values",
            config={"topic_type": "1"},  # 自定义一个config
    ):
        step['messages'][-1].pretty_print()  # 打印

