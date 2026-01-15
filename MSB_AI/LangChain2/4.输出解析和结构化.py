from llm_libs import llm
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
import json

# 1、方式一：使用 StrOutputParser
chain =  llm | StrOutputParser()
resp = chain.invoke(input="中国最后一个皇帝是谁？")
print(resp)

# 2、方式2：使用 SimpleJsonOutputParser(deepseek可能会不支持）
# 创建聊天提示模板，要求模型以特定格式回答问题
# prompt = ChatPromptTemplate.from_template(
#     "尽你所能回答用户的问题。"  # 基本指令
#     '你必须始终输出一个包含"answer"和"followup_question"键的JSON对象。其中"answer"代表：对用户问题的回答；"followup_question"代表：用户可能提出的后续问题'
#     "{question}"  # 用户问题占位符
# )
# chain = prompt | llm | SimpleJsonOutputParser()
# resp = chain.invoke({"question": "细胞的动力源是什么？"})
# print(resp)

# 3、方式3：使用 with_structured_output
# 使用pydantic定义一个类
# class Joke(BaseModel):
#     """笑话（搞笑段子）的结构类(数据模型类 POVO)"""
#     setup: str = Field(description="笑话的开头部分")  # 笑话的铺垫部分
#     punchline: str = Field(description="笑话的包袱/笑点")  # 笑话的爆笑部分
#     rating: Optional[int] = Field(description="笑话的有趣程度评分，范围1到10")  # 可选的笑话评分字段
# prompt_template = PromptTemplate.from_template("帮我生成一个关于{topic}的笑话。")
# runnable = llm.with_structured_output(Joke)
# chain = prompt_template | runnable
# resp = chain.invoke({"topic": "猫"})
# print(resp) # 结构化输出
# print(resp.__dict__) # 结构化输出的字典格式
# print(json.dumps(resp.__dict__)) # 结构化输出的字符串格式

# 4、方式4：使用 bind_tools （处理 deepseek 无法结构化输出问题，底层本质也是基于 with_structured_output 实现）
# class ResponseFormatter(BaseModel):
#     """始终使用此工具来结构化你的用户响应"""  # 文档字符串说明这个类用于格式化响应
#     answer: str = Field(description="对用户问题的回答")  # 回答内容字段
#     followup_question: str = Field(description="用户可能提出的后续问题")  # 后续问题字段
# runnable = llm.bind_tools([ResponseFormatter])
# resp = runnable.invoke("细胞的动力源是什么？")
# print(resp)
# print(resp.tool_calls[-1]['args'])
# resp.pretty_print() # 美化打印


