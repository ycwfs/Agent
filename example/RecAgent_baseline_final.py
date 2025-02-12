import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryBase
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import random
import re
import logging
import time
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class RecMemory(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type = "rec", llm=llm)

    def __call__(self, current_situation: str = ''):
        if 'item:' in current_situation:
            self.addMemory(current_situation.replace('item:', ''))
        else:
            return self.retriveMemory(current_situation)
    def retriveMemory(self, query_scenario: str):
        # Extract task name from query scenario
        task_name = query_scenario
        
        # Return empty string if memory is empty
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
            
        # Extract task trajectories from results
        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]
        
        # Join trajectories with newlines and return
        return '\n'.join(task_trajectories)

    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
                        The following are some examples:
                        Task: I need to find some information to complete a recommendation task.
                        sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
                        sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
                        sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

                        Task: {task_description}
                        '''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
                        The following are some examples:
                        Task: I need to find some information to complete a recommendation task.
                        sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
                        sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
                        sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

                        end
                        --------------------
                        Reflexion:{feedback}
                        Task:{task_description}
                        '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''{task_description}'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=4096
        )
        
        return reasoning_result

class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)
        self.memory = RecMemory(llm=self.llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # plan = self.planning(task_type='Recommendation Task',
        #                      task_description="Please make a plan to query user information, you can choose to query user, item, and review information, ",
        #                      feedback='',
        #                      few_shot='')
        # print(f"The plan is :{plan}")
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find review information'},
         {'description': 'Next, I need to find item information'}
         ]

        user = ''
        item_list = []
        history_item_reviews = []
        filtered_item_reviews = []
        history_reviews = []
        loc = self.task['loc']
        candidate_category = self.task['candidate_category']
        history_review = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
        similar_review = None

        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
            elif 'review' in sub_task['description']:
                # find all reviews from the user
                for review in history_review:
                    item_id = review['item_id']
                    item_statictis = self.interaction_tool.get_item(item_id=item_id)
                    self.memory(f'item:{item_statictis}')
            elif 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
                    similar_item = self.memory(str(item))
                    item_reviews = self.interaction_tool.get_reviews(item_id=self.task['candidate_list'][n_bus])
                    item_reviews = json.dumps(random.choice(item_reviews))

                    similar_review_id = eval(similar_item)['item_id']
                    for review in history_review:
                        if review['item_id'] == similar_review_id:
                            similar_review = json.dumps(review)

                    if similar_item == '':
                        history_item_reviews.append((item['item_id'] + ':' + item_reviews))
                    elif similar_item != '' and similar_review != None:
                        history_item_reviews.append((item['item_id'] + ':' + item_reviews + ', similar item you reviewed: ' + similar_review))
            else:
                pass
        # DO NOT output your analysis process!??????????
        # task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        #                         Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        #                         Please rank the more interested items more front in your rank list. The information of the above 20 candidate items is as follows: {item_list}.
        #                         Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        #                         The correct output format:['item id1', 'item id2', 'item id3', ...]
        # '''
    #     task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_reviews}. 
    # Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
    # Please rank the more interested items more front in your rank list. The information and reviews of the above 20 candidate items is as follows: {history_item_reviews}.
    # Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
    # The correct output format:['item id1', 'item id2', 'item id3', ...]
    #     '''

            task_description = f'''You are a real user on an online platform. 
    Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
    Please rank the more interesting items more front in your rank list. The information and reviews of the above 20 candidate items and their similar item are as follows: {history_item_reviews}.
    Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
    The correct output format:['item id1', 'item id2', 'item id3', ...]
        '''

#         task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
# Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
# Please rank the more interested items more front in your rank list. The information and average reviews score of the above 20 candidate items is as follows: {history_item_reviews}.
# Your final output should be ONLY a ranked item list with the following format: ['item id1', 'item id2', 'item id3', ...], DO NOT introduce any other item ids!
#         '''
        # memory = self.memory(task_description)
        # print("memory:", memory)
        # prompt_len = num_tokens_from_string(task_description)
        # if prompt_len > 4096:
        #     print(f'prompt_len: {prompt_len}')
        #print('task_description:',task_description)
        result = self.reasoning(task_description)

        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print('Meta Output:',result)
                print("No list found.")
            print('Processed Output:',eval(result))
            return eval(result)
        except:
            print('format error')
            return ['']


if __name__ == "__main__":
    task_set = "yelp" # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="/AgentSocietyChallenge/data", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"/AgentSocietyChallenge/example/track2/{task_set}/tasks", groundtruth_dir=f"/AgentSocietyChallenge/example/track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    simulator.set_llm(InfinigenceLLM(api_key="sk-dapxrd44nc6qjgxk"))
    #simulator.set_llm(OpenAILLM(api_key="sk-"))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
