"""
1、依赖安装：pip install --pre -U langchain langchain-openai
2、学习文档：https://www.langchain.com.cn/docs/how_to/
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

print("=" * 50)
print("1、内置大模型")
print("=" * 50)
def get_weather(city: str) -> str:
    """拿到一个城市的天气情况"""
    return f"{city}天气是大晴天!"
agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "北京的天气怎样？"}]}
)
print(f"内置大模型响应结果：{result}")



print("=" * 50)
print("2、未内置大模型")
print("=" * 50)
# 先创建一个模型实例
model = ChatOpenAI(
    model="glm-4",
    base_url=os.getenv("GLM_OPENAI_BASE_URL"),
    api_key=os.getenv("GLM_OPENAI_API_KEY"),
)
def get_weather(city: str) -> str:
    """拿到一个城市的天气情况"""
    return f"{city}天气是大晴天!"
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "上海的天气怎样？"}]}
)
print(f"未内置大模型响应结果：{result}")