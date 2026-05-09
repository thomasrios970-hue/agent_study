from Planner import Planner
from Executor import Executor
from LLM import HelloAgentsLLM
class PlanAndSolveAgent:
    def __init__(self,llm_client):
        #初始化智能体
        self.llm_client=llm_client
        self.planner=Planner(llm_client)
        self.executor=Executor(llm_client)

    def run(self,question:str)->str:
        #先规划、后执行

        print(f"\n----开始处理----\n问题{question}")

        plan=self.planner.plan(question)
        #调用规划器生成计划

        if not plan:
            print("\n--- 任务终止 ---\n无法生成有效的行动计划")
            return
        #检查计划是否成功生成

        final_answer=self.executor.execute(question,plan)

        return final_answer

if __name__=="__main__":
    planAndSolveAgent=PlanAndSolveAgent(HelloAgentsLLM())
    question="我现在在光明实验室，我要去剪头发怎么办，请你给我点建议"
    answer=planAndSolveAgent.run(question)
    print(f"\n最终答案：{answer}")