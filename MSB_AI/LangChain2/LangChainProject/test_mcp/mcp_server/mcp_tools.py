from fastmcp import FastMCP
from zhipuai import ZhipuAI

from env_utils import ZHIPU_API_KEY

zhipuai_client = ZhipuAI(api_key=ZHIPU_API_KEY)  # 填写您自己的APIKey
my_mcp_server = FastMCP(name='lx-test_mcp', instructions='我自己的MCP服务')

# 装饰器
@my_mcp_server.tool('my_search_tool', description='专门搜索互联网中的内容')
def my_search(query: str) -> str:
    """搜索互联网上的内容,包括实时天气等"""
    try:
        print("执行我的Python中的工具，输入的参数为:", query)
        response = zhipuai_client.web_search.web_search(
            search_engine="search_pro",
            search_query=query
        )
        # print(response)
        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
        return '没有搜索到任何内容！'
    except Exception as e:
        print(e)
        return '没有搜索到任何内容！'


# @my_mcp_server.tool()
# def add(a: int, b: int) -> int:
#     """加法运算: 计算两个数字相加"""
#     print(f'加法运算，传入的的参数是: ${a} 和  ${b}')
#     return a + b
#
#
# @my_mcp_server.tool()
# def multiply(a: int, b: int) -> int:
#     """乘法运算：计算两个数字相乘"""
#     print(f'乘法运算，传入的的参数是: ${a} 和  ${b}')
#     return a * b
