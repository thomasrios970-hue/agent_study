from typing import List,Dict,Any,Optional

class Memory:
    #短期记忆模块

    def __init__(self):
        #初始化空列表来存储所有记录
        self.records:List[Dict[str,Any]]=[]
    
    def add_record(self,record_type:str,content:Any):
        #添加记录

        record={"type":record_type,"content":content}
        self.records.append(record)
        print(f"已添加记录：{record}")

    def get_trajectory(self)->str:
        #将所有记忆记录化为一个连贯的字符串文本，用以构建提示词

        trajectory_parts=[]
        for record in self.records:
            if record["type"]=="execution":
                trajectory_parts.append(f"---上一轮尝试（代码）---\n{record['content']}")
            elif record["type"]=="reflection":
                trajectory_parts.append(f"---评审员反馈---\n{record['content']}")

        return "\n\n".join(trajectory_parts)
    
    def get_last_execution(self)->Optional[str]:
        #获取最近一次的执行结果

        for record in reversed(self.records):
            if record["type"]=="execution":
                return record["content"]
        return None
