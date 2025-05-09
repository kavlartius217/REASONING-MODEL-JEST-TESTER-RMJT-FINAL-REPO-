from crewai.flow import Flow, router, start, listen, and_, or_
from pydantic import BaseModel
from typing import Optional
import nest_asyncio
nest_asyncio.apply()

class State(BaseModel):
  test_code:str=""
  expected_coverage:int=0
  feedback:str=""
  pass_fail:str=""
  task_id_4:str=""
  task_id_5:str=""
    
class RMJT(Flow[State]):
    en_gen = EnhancedGenerator()
    """Reasoning Model Jest Tester"""

    @start()
    def code_gen(self):
        response = self.en_gen.crew().kickoff({"feedback": " "})
        self.state.test_code = self.en_gen.test_case_generator_task().output.raw
        self.state.feedback = response['feedback']
        self.state.pass_fail = response['pass_fail']
        self.state.expected_coverage = response['expected_coverage']

    @router(code_gen)
    def router_1(self):
        if self.state.expected_coverage < 90 and self.state.pass_fail == "FAIL":
            return "activate feedback mechanism"
        else:
            return "Passed"

    @listen("activate feedback mechanism")
    def task_ids(self):
        import subprocess
        x = subprocess.run(['crewai', 'log-tasks-outputs'],
                          capture_output=True, text=True, check=True).stdout.splitlines()
        for i in x:
            if 'Task 4:' in i:
                parts = i.split('Task 4:')
                if len(parts) > 1:
                    self.state.task_id_4 = parts[1].strip()
                    print(f"Found task 4 ID: {self.state.task_id_4}")
            
            if 'Task 5:' in i:
                parts = i.split('Task 5:')
                if len(parts) > 1:
                    self.state.task_id_5 = parts[1].strip()
                    print(f"Found task 5 ID: {self.state.task_id_5}")
        # Print both IDs for verification
        print(f"Task 4 ID: {self.state.task_id_4}")
        print(f"Task 5 ID: {self.state.task_id_5}")

    @listen(or_(task_ids,'re-run'))
    def code_gen_m2(self):
        task_id = self.state.task_id_4
        feedback = {"feedback": self.state.feedback}
        response = self.en_gen.crew().replay(task_id=task_id, inputs=feedback)
        print(response.raw)

    @listen(code_gen_m2)
    def static_testing_m2(self):
        task_id = self.state.task_id_5
        response = self.en_gen.crew().replay(task_id=task_id)
        self.state.feedback = response['feedback']
        self.state.pass_fail = response['pass_fail']
        self.state.expected_coverage = response['expected_coverage']
        

    @router(static_testing_m2)
    def router_2(self):
      if self.state.expected_coverage < 90 and self.state.pass_fail == "FAIL":
            return "re-run"
      else:
            return "Test Cases Passed"

    
    @listen("Test Cases Passed")
    def show(self):
      print(self.state.expected_coverage)
      print(self.state.pass_fail)
