"""
知识点：
一、创建 Langgraph 项目
1、创建虚拟环境
（1）创建虚拟环境：python -m venv .venv
（2）激活虚拟环境：.\.venv\Scripts\activate
（3）退出虚拟环境：deactivate
（2）python 版本要大于 3.11
2、安装脚本 langgraph-cli
（1）pip install --upgrade "langgraph-cli[inmem]"
（2）inmem 是可选功能，英文是 in-memory，作用是在开发阶段增加内存存储的能力，而不用自己添加
3、创建项目模板
（1）不使用模板（交互式命令选择）：langgraph new
（2）使用模板：langgraph new my_langgraph --template new-langgraph-project-python
"""