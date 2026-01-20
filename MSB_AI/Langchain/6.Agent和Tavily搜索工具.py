"""
1、依赖包：pip install -U langchain-tavily
2、浏览器搜索 tavily-api-key，获取 key: https://app.tavily.com/home  [环境变量名：TAVILY_API_KEY]
"""

"""
前置知识点：
1、大模型无法获取到实时的数据，因为训练大模型的数据可能是几个月或者几年之前的
2、通过代理的方式，可以让大模型去搜索实时数据，并返回给用户
3、大模型能自动判断问题自己能回答，还是不能回答（不能回答就主动通过代理去搜索，代理可以设置多个）
4、在企业中实践时，先判断模型返回的数据是否为空，如果为空，就证明模型无法回答该问题，那么直接拿代理的数据就可以
5、resp['messages'] 是一个数组：
（1）模型回答的数据，数组长度为2，模型的数据为 resp['messages'][1].content
（2）代理回答的数据，数组的长度至少为3，第一个代理的数据为 resp['messages'][2].content
"""
import warnings
# 忽略所有来自 langchain_tavily 模块的 UserWarning(命名重复了）
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

# 配置环境变量
load_dotenv()

def setup_agent():
    """设置并返回配置好的代理"""
    # 验证API密钥
    if not os.environ.get("TAVILY_API_KEY"):
        raise ValueError("请设置TAVILY_API_KEY环境变量")

    # 创建模型
    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.1  # 降低随机性，使回答更稳定
    )

    # 创建搜索工具
    search_tool = TavilySearch(
        max_results=3,  # 增加结果数量
        search_depth="advanced"  # 更深入的搜索
    )

    tools = [search_tool]

    # 创建代理
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个有用的助手，可以搜索实时信息来回答问题。"
    )

    return agent


def ask_question(agent, question):
    """向代理提问并返回结构化结果"""
    try:
        response = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })

        messages = response['messages']
        result = {
            'question': question,
            'message_count': len(messages),
            'used_tools': len(messages) > 2  # 是否使用了工具
        }

        if result['used_tools']:
            result['tool_result'] = messages[-2].content
            result['final_answer'] = messages[-1].content
        else:
            result['final_answer'] = messages[-1].content

        return result

    except Exception as e:
        return {'error': str(e), 'question': question}


# 使用示例
if __name__ == "__main__":
    agent = setup_agent()

    # 测试不同问题
    questions = [
        "中国的首都是哪个城市？",  # 常识问题
        "今天北京的最高温度是多少？",  # 实时信息
        "特斯拉股票今天的最新价格是多少？"  # 实时金融信息
    ]

    for question in questions:
        print(f"\n问题: {question}")
        result = ask_question(agent, question)

        if 'error' in result:
            print(f"错误: {result['error']}")
        else:
            print(f"消息数量: {result['message_count']}")
            print(f"使用工具: {'是' if result['used_tools'] else '否'}")
            if result['used_tools']:
                print(f"工具结果: {result['tool_result'][:100]}...")
            print(f"最终回答: {result['final_answer']}")
