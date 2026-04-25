import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
class HelloAgentsLLM:
    def __init__(self,model:str=None,apiKey:str=None,baseUrl:str=None,timeout:int=None):
        #初始化模型参数
        self.model=os.environ.get("LLM_MODEL_ID")
        self.apiKey=os.environ.get("OPENAI_API_KEY") or apiKey
        self.baseUrl=os.environ.get("LLM_BASE_URL")
        self.timeout=timeout
        #设置模型参数

        if not all([self.model,self.apiKey,self.baseUrl]):
            raise ValueError("模型的API密钥、baseurl或者名称没读取成功")

        self.client=OpenAI(api_key=self.apiKey,base_url=self.baseUrl,timeout=self.timeout)
    #------------------------------------------------------------------
    def think(self,messages:List[Dict[str,str]],temperature:float=0)->str:
        #调用大语言API，如firts_agent里面的LLM.py里的generate()
        print(f"正在调用{self.model}模型...")
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )        #处理流式响应
            print("大语言模型响应成功")
            collected_content=[]
            for chunk in response:
                content=chunk.choices[0].delta.content
                print(content,end='',flush=True)
                collected_content.append(content)

            print()#输出完换行，因为之前的把end换成了""
            return "".join(collected_content)
        except Exception as e:
            print(f"调用LLM.py的时候发生错误{e}")
            return None

if __name__=='__main__':
    try:
        llm=HelloAgentsLLM()

        messages=[{"role":"system","content":"你是一个对编写代码有用的代码助手"},
                  {"role":"user","content":"编写一个归并排序的代码"}]

        print("-----------调用大模型中----------------")
        response=llm.think(messages)
        if response:
            print("\n\n完整模型响应如下：")
            print(response)
    except Exception as e:
        print(e)







