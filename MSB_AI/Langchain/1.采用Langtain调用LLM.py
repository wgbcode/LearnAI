from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 步骤一：创建大模型实例
model = ChatOpenAI(model='gpt-5-mini')

# 步骤二：创建提示模板（修复关键）
prompt = ChatPromptTemplate.from_messages([
    ("system", "请将以下的内容翻译成英语"),
    ("human", "{text}")
])
# 生成的数据结构示例：
# [
#     SystemMessage(content="请将以下的内容翻译成英语"),
#     HumanMessage(content="你好")
# ]

# 步骤三：正确实例化输出解析器（修复关键）
parser = StrOutputParser()  # 注意这里需要括号！

# 步骤四：构建正确的处理链
chain = prompt | model | parser

# 步骤五：使用链式调用（传递字典参数）
result = chain.invoke({"text": "你好，请问你要去哪里？"})
print("翻译结果:", result)


















