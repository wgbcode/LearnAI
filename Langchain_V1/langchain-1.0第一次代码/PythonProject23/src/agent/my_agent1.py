from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agent.my_llm import llm


# def send_email(to: str, subject: str, body: str):
#     """发送邮件"""
#     email = {
#         "to": to,
#         "subject": subject,
#         "body": body
#     }
#     # ... 邮件发送逻辑
#
#     return f"邮件已发送至 {to}"

@tool('calculate', parse_docstring=True)
def calculate4(a: float, b: float, operation: str):
    """工具函数：计算两个数字的运行结果

    Args:
        a: 第一个数字
        b: 第二个数字
        operation: 运算类型，只支持加减乘除

    Returns:
        float: 运行结果

    """

    print(f"调用 calculate 工具，第一个数字：{a}, 第二个数字：{b}, 运算类型：{operation}")

    result = 0.0
    match operation:
        case "add":
            result = a + b
        case "加":
            result = a + b
        case "subtract":
            result = a - b
        case "multiply":
            result = a * b
        case "divide":
            if b != 0:
                result = a / b
            else:
                raise ValueError("除数不能为零")
    print(f"计算结果：{result}")
    return result




# prompt_template = PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的描述")
prompt_template = (PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的描述") + "，要求：1、搞笑一点"+"；2、使用{language}")



class ChainToolsArgs(BaseModel):
    topic: str = Field(..., description="报幕词的主题")
    language: str= Field(..., description="报幕词使用的语言")


chain = prompt_template | llm

runnable_tool = chain.as_tool(
    name="报幕词生成",
    description="这是一个专门生成报幕词的工具",
    args_schema=ChainToolsArgs
)


agent = create_agent(
    llm,
    tools=[runnable_tool],
    system_prompt="你是一个邮件助手。请始终使用 send_email 工具。",
)