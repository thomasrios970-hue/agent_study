from LLM import HelloAgentsLLM
from ToolExecutor import ToolExecutor
from Tools import search
import re
# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self,llm_client:HelloAgentsLLM,tool_executor:ToolExecutor,max_steps:int=5):
        self.llm_client=llm_client
        self.tool_executor=tool_executor
        self.max_steps=max_steps
        self.history=[]

    def run(self,question:str)->str:
        #核心运行逻辑

        self.history=[]
        #每次运行重置历史记录
        current_step=0

        while current_step<self.max_steps:
            current_step+=1
            print(f"\n---第{current_step}步---")

            #格式化提示词
            tools_des=self.tool_executor.getAvailableTools()
            history_str="\n".join(self.history)
            prompt=REACT_PROMPT_TEMPLATE.format(
                tools=tools_des,
                question=question,
                history=history_str
            )

            #调用LLM进行思考
            messages=[{"role":"user","content":prompt}]
            response_text=self.llm_client.think(messages)

            if not response_text:
                print( "LLM没有返回任何内容，结束运行")
                break
        
        #工具调用与执行
            thought,action=self._parse_response(response_text)

            if thought:
                print(f"思考（Thought）：{thought}")

            if not action:
                print("没有解析到Action，结束运行")
                break

            #执行Action
            if action.startswith("Finish"):
                final_answer=re.match(r"Finish\[(.*)\]",action,re.DOTALL).group(1)
                print(f"最终答案：{final_answer}")
                return final_answer
            
            tool_name,tool_input=self._parse_action(action)
            if not tool_name or not tool_input :
                continue
            #处理无效Action格式

            print(f"行动：{tool_name}[{tool_input}]")

            tool_function=self.tool_executor.getTool(tool_name)
            if not tool_function:
                print(f"错误：未找到名为{tool_name}的工具")
                continue
            else:
                observation=tool_function(tool_input)

            print(f"观察：{observation}")

            #将本轮的Action、Observation添加到历史记录中
            self.history.append(f"Action:{action}")
            self.history.append(f"Observation:{observation}")

        print("已达到最大步数，流程终止")
        return None



    
    #输出解析器实现
    def _parse_response(self,text:str):
        #解析LLM的输出，提取Thought和Action

        thought_match=re.search(r"Thought:\s*(.*?)(?=Action:|$)",text,re.DOTALL)
        #Thought:匹配到Action：或文本结尾

        action_match=re.search(r"Action:\s*(.*?)$",text,re.DOTALL)
        
        thought=thought_match.group(1).strip() if thought_match else ""
        action=action_match.group(1).strip() if action_match else ""
        return thought,action
    
    def _parse_action(self,action_text:str):
        #解析Action文本，提取工具名称和输入
        
        match=re.match(r"(\w+)\[(.*)\]",action_text,re.DOTALL)
        if match:
            return match.group(1),match.group(2)
        return None,None

if __name__=="__main__":
    toolExecutor=ToolExecutor()
    search_description="一个网页搜索引擎。当你需要回答时事、事实以及在你的知识库中找不到信息时，应使用此工具。"
    toolExecutor.register_tool("Search",search_description,search)

    reActAgent=ReActAgent(HelloAgentsLLM(),toolExecutor)
    question="深圳的特产是什么"
    answer=reActAgent.run(question)
    print(f"\n最终回答：{answer}")
