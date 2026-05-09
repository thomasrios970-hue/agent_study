from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from define_agent import *
from create_LLM import create_openai_model_client

model_client=create_openai_model_client()
product_manager=create_product_manager(model_client)
engineer=create_engineer(model_client)
code_reviewr=create_code_reviewer(model_client)
user_proxy=create_user_proxy()

# 定义团队聊天和协作规则
team_chat=RoundRobinGroupChat(
    participants=[
        product_manager,
        engineer,
        code_reviewr,
        user_proxy
    ],
    # 决定了智能体发言的先后顺序
    termination_condition=TextMentionTermination("TERMINATE"),
    # 设定当出现关键词TERMINATE时，对话便终止
    max_turns=20
    # 设置最大轮次，到最大轮次就终止
)

