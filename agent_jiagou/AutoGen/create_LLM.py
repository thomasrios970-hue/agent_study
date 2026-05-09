from autogen_ext.models.openai import OpenAIChatCompletionClient
# AutoGen提供了标准化的OpenAIChatCompletionClient，可以方便的与任何兼容OpenAI API规范的模型服务进行对接
import os

def create_openai_model_client():
    """创建并配置OpenAI模型客户端"""
    return OpenAIChatCompletionClient(
        model="qwen3.6-plus",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.gpt.ge/v1" ,
        model_info={
            "function_calling": True,    # 如果模型支持函数调用
            "vision": True,              # 如果模型支持图片输入
            "json_output": True,         # 如果模型支持 JSON 模式
            "family": "unknown",         # 或填 "qwen" 等标识
        }
    )

