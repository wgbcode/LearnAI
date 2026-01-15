"""
1、依赖：pip install langchain-chroma
"""

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# 引入环境变量
load_dotenv()

# 创建模型
model = ChatOpenAI(model='gpt-5-mini')

# 准备测试数据，假设有5份文档
documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

"""
构建一个向量空间
1、使用OpenAIEmbeddings引擎，将测试文档内的文本转换成一系列数字，即向量（每个字符对应一个数字）
2、将转换后的数据，保存到向量数据库 Chroma
3、拿到一个可以操作的向量数据库实例
"""
vector_store = Chroma.from_documents(documents,embedding=OpenAIEmbeddings())

# 相似度查询。加“_with_score”的意思是把“向量相似度评分”也返回，分数越低相似度越高
# print(vector_store.similarity_search_with_score('咖啡猫'))

# 将 vector_store 转换成一个 Runnable 对象，这样才能加到 Chain 链中
# ".bind(k=1)"的作用：只返回相似度最高的一个文档
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

# 同时查询两个相似度，返回的文档也会是两个
# print(retriever.batch(['咖啡猫', '鲨鱼']))

"""
将向量空间和模型结合起来使用
1、向量空间为模型提供上下文（外部数据）
2、模型根据上下文回答用户的问题
"""

# 先创建一个提问模板
message = """
请根据上下文回答我的问题：
问题：{question}
上下文：{context}
"""

"""
提示模板角色的作用：
1、system:系统角色，定义模型的身份、行为准则和任务背景，为整段对话设定基调和规则
2、user和human：用户角色，代表用户的输入
"""
prompt_temp = ChatPromptTemplate.from_messages([
    ('human',message)
])

# 构建链
# RunnablePassthrough 的作用：允许我们在后面再传入具体的 prompt 和 model
chain = {"question":RunnablePassthrough(),"context":retriever} | prompt_temp | model

print(chain.invoke("猫是什么？"))











