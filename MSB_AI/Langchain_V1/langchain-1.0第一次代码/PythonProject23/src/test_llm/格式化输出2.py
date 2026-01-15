from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.my_llm import llm

# 创建聊天提示模板，要求模型以特定格式回答问题
prompt = ChatPromptTemplate.from_template(
    "尽你所能用中文，回答用户的问题。"  # 基本指令
    '你必须始终输出一个包含"title","year","director","rating"键的JSON对象。其中"title"代表：电影标题；"year"代表：电影发行年份；"director"代表：电影导演的名字；"rating"代表：电影评分（满分10分）。'
    "{question}"  # 用户问题占位符
)


chain = prompt | llm | SimpleJsonOutputParser()
resp = chain.invoke({"question": "提供电影《盗梦空间》的详细信息？"})
print(resp)