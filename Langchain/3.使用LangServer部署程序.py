
# 第三方依赖包：pip install "langserve[all]"
# 接口测试工具：Postman； ApiPost

import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes, RemoteRunnable

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
# print(chain.invoke({
#     'language': 'English',
#     'text':'杰克，我下午还有一节课，不能去打蓝球了'
# }))

# 步骤六：把我们的程序部署成服务
# 创建 fastAPI 的应用
app = FastAPI(title='我的Langchain服务',version='1.0.0',description='使用Langchain翻译任何语句的服务')
# 添加路由
add_routes(app,chain,path='/chainDemo')
# 启动服务
if __name__ == '__main__':
    # post 请求
    # 接口路径：http://127.0.0.1:8000/chainDemo/invoke
    # 传参：Body，application/json。{"input":{"language":"English","text":"我要去上课了,不能和你聊天了"}}
    # 输出：output
    uvicorn.run(app, host='127.0.0.1', port=8000)


# 不使用 postman，自己写代码测试(另写一个client文件运行测试）3.使用LangServer部署程序.py
# if __name__ == '__main__':
#     client = RemoteRunnable("http://172.0.0.1:8000/chainDemo/")
#     print(client.invoke({"language":"English","text":"我要去上课了,不能和你聊天了"}))

















