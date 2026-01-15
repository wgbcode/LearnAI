import os

from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI


# 聊天机器人案例
# 创建模型
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8)

# 创建链
chain = create_data_generation_chain(model)

# 生成数据
# result = chain(  # 给于一些关键词， 随机生成一句话
#     {
#         "fields": ['蓝色', '黄色'],
#         "preferences": {}
#     }
# )

result = chain(  # 给于一些关键词， 随机生成一句话
    {
        "fields": {"颜色": ['蓝色', '黄色']},
        "preferences": {"style": "让它像诗歌一样。"}
    }
)
print(result)

