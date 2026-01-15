import re

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


loader = RecursiveUrlLoader(
    "https://docs.python.org/zh-cn/3.13/tutorial/controlflow.html",
    max_depth=2,  # 可选参数：最大递归深度，默认为None表示不限制
    # exclude_dirs=(
    #     'https://docs.python.org/zh-cn/3/library/index.html',
    #     'https://docs.python.org/zh-cn/3/reference/index.html',
    #     'https://docs.python.org/zh-cn/3/extending/index.html',
    #     'https://docs.python.org/zh-cn/3/c-api/index.html',
    # ),  # 可选参数：要排除的目录路径元组，默认为空
    # base_url='https://docs.python.org/',  # 可选参数：基础URL，用于解析相对链接，默认为None
    extractor=bs4_extractor
)
# Docs = loader.load()
# print(Docs[0])

pages = []
for doc in loader.lazy_load():
    pages.append(doc)

print(pages)