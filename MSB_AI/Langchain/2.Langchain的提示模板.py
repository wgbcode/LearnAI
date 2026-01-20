import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 步骤一：创建大模型实例
model = ChatOpenAI(model='gpt-5-mini')

# 步骤二：定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system','请将下面的内容翻译成{language}'),
    ('user',"{text}")
])

# 步骤三：创建返回的数据解析器
parser = StrOutputParser()

# 步骤四：得到链
chain = prompt_template | model | parser

# 步骤五：直接使用 chain 来调用
print(chain.invoke({
    'language': 'English',
    'text':'我下午还有一节课，不能去打球了'
}))


















