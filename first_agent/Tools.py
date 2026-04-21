import requests
import json
#get_weather所用库

import os
from tavily import TavilyClient
#search_attraction所用库

#查询真实天气
def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API查询真实的天气信息
    """
    url=f"https://wttr.in/{city}?format=j1"

    try:
        response=requests.get(url)
        #向网站发送get请求
        response.raise_for_status()
        #检查请求是否发送成功，不成功则终止程序
        data=response.json()
        #将响应体中的内容解析成python的字典或列表结构

        # print("json格式如下所示:", json.dumps(data, indent=4, ensure_ascii=False))
        # #观察格式

        current_condition=data["current_condition"][0]
        weather_desc=current_condition["weatherDesc"][0]["value"]
        temp_c=current_condition["temp_C"]
        #提取当前天气状况

        return f"{city}当前天气{weather_desc}，气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        #处理网络错误
        return f"错误：查询天气时遇到网络问题- {e}"
    except (KeyError,IndexError) as e:
        return f"错误：解析天气数据失败，可能是城市名称无效- {e}"

def get_attraction(city: str,weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """

    #1. 从环境变量中读取API密钥
    # 原因是因为密钥直接放在代码上是不安全的
    api_key=os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY环境变量"

    #2. 初始化Tavily客户端
    tavily=TavilyClient(api_key=api_key)

    #3. 构建一个精确的查询
    query=f"'{city}'在'{weather}'下最值得去的旅游景点推荐及理由"

    try:
        #4. 调用API,include_answer=True会返回一个综合性的回答
        response=tavily.search(query=query,search_depth="basic",include_answer=True)

        #5. Tavily返回的结果已经十分干净，可以直接使用
        # response['answer']是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        # 如果没用综合性回答，则格式化原始结果
        formatted_results=[]
        for result in response["results"]:
            formatted_results.append(f"-{result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐"

        return "根据搜索，为您找到以下信息：\n"+"\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行Tavily搜索时出现问题-{e}"
    


