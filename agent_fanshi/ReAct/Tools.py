from serpapi import SerpApiClient
from dotenv import load_dotenv
import os
import json

load_dotenv()

def search(query:str)->str:
    #搜索工具
    print(f"正在使用搜索工具搜索：{query}")
    try:
        api_key=os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            return "搜索工具的API密钥没有设置成功"
        
        client=SerpApiClient({
            "api_key":api_key,
            "engine":"google",
            "q":query,
            "gl":"cn",
            "hl":"zh-CN",
        })

        #api的参数就要查官网
        result=client.get_dict()

        # 从可能不完整的API响应中尽可能提取出最有价值的答案，并按清晰度排序
        if "answer_box_list" in result:
            return "\n".join(
                item if isinstance(item,str) else json.dumps(item,ensure_ascii=False)
                for item in result["answer_box_list"]
            )
        if "answer_box" in result and "answer" in result["answer_box"]:
            return result["answer_box"]["answer"]
        if "knowledge_graph" in result and "description" in result["knowledge_graph"]:
            return result["knowledge_graph"]["description"]
        if "organic_results" in result and result["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(result["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"对不起，没有找到关于 '{query}' 的信息。"
    except Exception as e:
        return f"调用搜索工具的时候发生错误：{e}"


