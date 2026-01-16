import os
from langchain.chat_models.base import BaseChatModel
from langchain.messages import (
    AIMessage, HumanMessage, SystemMessage)
from typing import List, Optional, Any, Dict, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGenerationChunk, ChatGeneration
from openai import OpenAI


class BailianCustomChatModel(BaseChatModel):
    """自定义聊天模型，直接返回AIMessage"""

    # 模型配置参数
    model_name: str = "deepseek-v3.2-exp"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    enable_thinking: bool = True
    temperature: float = 0.5
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return f"bailian_chat_{self.model_name}"

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """核心生成方法，返回ChatResult"""

        # 将LangChain消息转换为OpenAI格式
        openai_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                openai_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                openai_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                openai_messages.append({"role": "system", "content": message.content})

        # 调用API。还是使用 OpenAI 来调用
        client = OpenAI(
            api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.base_url,
        )

        call_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        if self.model_name in ["deepseek-v3.2-exp", "deepseek-v3.1"] and self.enable_thinking:
            call_params["extra_body"] = {"enable_thinking": True}

        try:
            completion = client.chat.completions.create(**call_params)
            rc = None
            if completion.choices and completion.choices[0].message.content:
                if getattr(completion.choices[0].message, 'model_extra', None):
                    # 核心代码，拿到深度思考的内容，并包装成 AIMessage 返回
                    if 'reasoning_content' in getattr(completion.choices[0].message, 'model_extra', None):
                        rc = getattr(completion.choices[0].message, 'model_extra', None)['reasoning_content']
                # 创建AIMessage
                aimessage = AIMessage(
                    content=completion.choices[0].message.content,
                    additional_kwargs={
                        "model": self.model_name,
                        "usage": getattr(completion, 'usage', {}),
                        "reasoning_content": rc if rc else ""
                    }
                )

                # 创建ChatResult
                return ChatResult(generations=[ChatGeneration(message=aimessage)])
            else:
                raise ValueError("模型返回空响应")

        except Exception as e:
            raise ValueError(f"调用百炼模型失败: {str(e)}")

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式生成方法，返回AIMessageChunk"""

        # 将LangChain消息转换为OpenAI格式
        openai_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                openai_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                openai_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                openai_messages.append({"role": "system", "content": message.content})

        # 调用API
        client = OpenAI(
            api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.base_url,
        )

        # 构建流式调用参数
        call_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        # 添加思考模式参数
        if self.model_name in ["deepseek-v3.2-exp", "deepseek-v3.1"] and self.enable_thinking:
            call_params["extra_body"] = {"enable_thinking": True}

        try:
            completion = client.chat.completions.create(**call_params)

            reasoning_content = ""  # 累积思考内容
            answer_content = ""  # 累积回复内容
            is_answering = False  # 是否进入回复阶段

            for chunk in completion:
                # 处理使用量统计信息等非内容块
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 处理思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    current_reasoning = delta.reasoning_content
                    reasoning_content += current_reasoning

                    # 生成思考内容的AIMessageChunk
                    reasoning_chunk = AIMessageChunk(
                        content="",  # 思考内容不作为主内容
                        additional_kwargs={
                            "reasoning_content": current_reasoning,
                            "reasoning_content_accumulated": reasoning_content
                        },
                        response_metadata={
                            "model_provider": "deepseek",
                            "chunk_type": "reasoning"
                        }
                    )

                    generation_chunk = ChatGenerationChunk(message=reasoning_chunk)

                    # 触发回调
                    if run_manager:
                        run_manager.on_llm_new_token(
                            current_reasoning,
                            chunk=generation_chunk
                        )

                    yield generation_chunk

                # 处理回复内容
                if hasattr(delta, "content") and delta.content is not None:
                    current_content = delta.content
                    answer_content += current_content

                    if not is_answering:
                        is_answering = True

                    # 生成回复内容的AIMessageChunk
                    content_chunk = AIMessageChunk(
                        content=current_content,
                        additional_kwargs={
                            "reasoning_content": reasoning_content if reasoning_content else ""
                        },
                        response_metadata={
                            "model_provider": "deepseek",
                            "chunk_type": "content"
                        }
                    )

                    generation_chunk = ChatGenerationChunk(message=content_chunk)

                    # 触发回调
                    if run_manager:
                        run_manager.on_llm_new_token(
                            current_content,
                            chunk=generation_chunk
                        )

                    yield generation_chunk

            # 流结束信号 - 发送一个空的chunk表示流结束
            end_chunk = AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning_content": reasoning_content,
                    "is_final": True
                },
                response_metadata={
                    "model_provider": "deepseek",
                    "chunk_type": "end_of_stream"
                }
            )
            yield ChatGenerationChunk(message=end_chunk)

        except Exception as e:
            # 错误处理chunk
            error_chunk = AIMessageChunk(
                content="",
                additional_kwargs={
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                response_metadata={
                    "model_provider": "deepseek",
                    "chunk_type": "error"
                }
            )
            yield ChatGenerationChunk(message=error_chunk)
            raise ValueError(f"流式调用百炼模型失败: {str(e)}")
