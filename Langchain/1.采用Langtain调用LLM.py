import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 步骤一：创建大模型实例
model = ChatOpenAI(model='gpt-4-turbo')

# 步骤二：创建提示
msg = [
    SystemMessage(content='请将以下的内容翻译成意大利语'),
    HumanMessage(content="你好，请问你要去哪里？")
]

# 不定义模板
result = model.invoke(msg)
print(result)


# 步骤三：创建返回的数据解析器
parser = StrOutputParser
# print(parser.invoke(result))

# 步骤四：得到链
chain = model | parser

# 步骤五：直接使用 chain 来调用
print(chain.invoke(msg))


















