"""
知识点：
1、示例一：通过原生方法调用 openai 的 Embedding 模型
2、示例二：通过 sentence_transformers 框架使用 Qwen3 的 Embedding 模型（推荐）
3、示例三：通过 langchain_huggingface 内置库使用 bge 的 Embedding 模型
4、示例四：在 langchain 中使用 Qwen3 的 Embedding 模型（未内置，需要自定义类）
5、示例五：通过 Embedding 模型，将本地原始数据转换为向量数据，并计算用户输入数据与每个向量数据的距离（相关性和语义）
"""

import ast
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

# 1、示例一：通过原生方法调用 openai 的 Embedding 模型
# openai_embedding = OpenAIEmbeddings(
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
#     model="text-embedding-3-large",
#     dimensions=2560,
# )
# resp = openai_embedding.embed_documents(
#     ['I like large language models.',
#      '今天的天气非常不错！'
#      ]
# )
# print(resp[0])
# print(len(resp[0]))


# 2、示例二：通过 sentence_transformers 框架使用 Qwen3 的 Embedding 模型
# """
# 1、需要安装： pip install sentence-transformers
# 2、Requires transformers>=4.51.0
# 3、Requires sentence-transformers>=2.7.0
# 4、需要翻墙或者使用镜像下载 Embedding 模型 （默认从 HF 下载模型）
# 5、环境变量：HF_ENDPOINT=https://hf-mirror.com
# 6、修改环境变量后，需要重启 pycharm
# """
# qwen3_embedding = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# resp = qwen3_embedding.encode(
#     ['I like large language models.',
#      '今天的天气非常不错！'
#      ]
# )
# print(resp[0])
# print(len(resp[0]))

# 3、示例三：通过 langchain_huggingface 内置库使用 bge 的 Embedding 模型
# """
# 1、需要安装：pip install langchain-huggingface
# 2、第一次运行，会自动下载模型（去huggingface上下载），下载到hf默认的缓存目录
# 3、可以通过修改环境变量：HF_HOME=指定目录
# """
# model_name = "BAAI/bge-small-zh-v1.5"
# model_kwargs = {'device': 'cpu'} # 使用 cpu 运行模型
# encode_kwargs = {'normalize_embeddings': True}
# bge_hf_embedding = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )
# resp = bge_hf_embedding.embed_documents(
#     ['I like large language models.',
#      '今天的天气非常不错！'
#      ]
# )
# print(resp[0])
# print(len(resp[0]))

# 4、示例四：在 langchain 中使用 Qwen3 的 Embedding 模型（未内置，需要自定义类）
# class CustomQwen3Embeddings(Embeddings):
#     """自定义一个qwen3的Embedding和langchain整合的类"""
#     def __init__(self, model_name):
#         self.qwen3_embedding = SentenceTransformer(model_name)
#
#     def embed_query(self, text: str) -> list[float]:
#         return self.embed_documents([text])[0]
#
#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         return self.qwen3_embedding.encode(texts)
#
# if __name__ == '__main__':
#     qwen3 = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")
#     resp = qwen3.embed_documents(
#         ['I like large language models.',
#          '今天的天气非常不错！'
#          ]
#     )
#     print(resp[0])
#     print(len(resp[0]))


# 5、示例五：通过 Embedding 模型，将本地原始数据转换为向量数据，并计算用户输入数据与每个向量数据的距离（相关性和语义）
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
bge_hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# 将文本转换成向量数据
def text_2_embedding(text):
    resp = bge_hf_embedding.embed_documents(
        [text]
    )
    return resp[0]
# 生成向量数据，并保存成一个新的文件存放
def embedding_2_file(source_file, output_file):
    """读取原始的美食评论数据，通过调用Embedding模型，得到向量，并保持到新文件中"""
    #  步骤：1、准备数据，并读取
    df = pd.read_csv(source_file, index_col=0)
    df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]
    print(df.head(2))
    # 步骤2： 清洗数据和合并数据
    # df = df.dropna()
    # 把评论的摘要和内容字段合并成 一个字段（方便后续处理）
    df['text_content'] = 'Summary: ' + df.Summary.str.strip() + "; Text: " + df.Text.str.strip()
    print(df.head(2))  # 增加一个text_content
    # 步骤3: 向量化，存到一个新的文件中
    df['embedding'] = df.text_content.apply(lambda x: text_2_embedding(x))
    df.to_csv(output_file)
# 计算提问问题（已转换成向量数据）与每一行向量数据的余弦值（后面需要排序）
def cosine_distance(a, b):
    """计算余弦距离"""
    return  np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# 在内存中计算，并输出结果（不需要输出为文件——
def search_text(input, embedding_file, top_n=3):
    """
    根据用户输入的问题，进行语义检索，返回最相似的前top_n个结果
    :param input:
    :param top_n:
    :return:
    """
    df_data = pd.read_csv(embedding_file)
    # 把字符串变成向量，保持到新字段
    df_data['embedding_vector'] = df_data['embedding'].apply(ast.literal_eval)
    input_vector = text_2_embedding(input)
    df_data['similarity'] = df_data.embedding_vector.apply(lambda x: cosine_distance(x, input_vector))
    res = (
        df_data.sort_values('similarity', ascending=False)
        .head(top_n)
        .text_content.str.replace('Summary: ', "")  # text_content是字段名
        .str.replace('; Text: ', ';')
    )
    for r in res:
        print(r)
        print('-' * 30)

if __name__ == '__main__':
    # TODO:这一行代码只需要在初始化时执行一次
    # embedding_2_file('./datas/embedding/fine_food_reviews_1k.csv', './datas/embedding/output_embedding.csv')
    search_text('I like juicy barbecued meat.', 'datas/embedding/output_embedding.csv')





