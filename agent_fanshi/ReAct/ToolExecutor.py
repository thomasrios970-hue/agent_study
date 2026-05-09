from typing import Dict,Any

class ToolExecutor:
    #工具执行器，负责管理和执行工具
    def __init__(self):
        self.tools:Dict[str,Dict[str,Any]]={}
    
    def register_tool(self,name:str,description:str,func:callable):
        #注册工具
        if name in self.tools:
            print(f"工具{name}已经注册过了，覆盖之前的工具")
        self.tools[name]={"description":description,"func":func}
        print(f"工具 {name} 注册成功")

    def getTool(self,name:str)->callable:
        #根据名称获取一个工具的执行函数
        tool=self.tools.get(name)
        if tool:
            return tool.get("func")

        normalized_name=name.lower()
        for registered_name,info in self.tools.items():
            if registered_name.lower()==normalized_name:
                return info.get("func")
        return None
    
    def getAvailableTools(self)->str:
        #获取所有可用工具的格式化描述字符串
        return "\n".join([f"-{name}:{info['description']}" 
                          for name,info in self.tools.items()])

#使用示例
if __name__=="__main__":
    toolExecutor=ToolExecutor()
    #初始化工具执行器

    from Tools import search
    search_description="一个网页搜索引擎。当你需要回答时事、事实以及在你的知识库中找不到信息时，应使用此工具"
    toolExecutor.register_tool("Search",search_description,search)
    #注册搜索工具
    
    print("\n----当前可用工具---")
    print(toolExecutor.getAvailableTools())
    #打印可用的工具

    print("\n---执行Action：Search['英伟达最新的GPU是什么？']---")
    tool_name="Search"
    tool_input="英伟达最新的GPU是什么？"

    tool_function=toolExecutor.getTool(tool_name)
    if tool_function:
        observation=tool_function(tool_input)
        print("---观察（observation）---")
        print(observation)
    else:
        print(f"错误：未找到名为{tool_name}的工具")
