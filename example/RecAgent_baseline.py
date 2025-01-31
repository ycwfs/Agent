import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryBase
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
        #self.memory = RecMemory(llm=self.llm)

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
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user = ''
        item_list = []
        history_item_reviews = []
        history_review = ''
        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
                    keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                    #item_review = self.interaction_tool.get_reviews(item_id=self.task['candidate_list'][n_bus])
                    #history_item_reviews.append((str(filtered_item) + str(item_review)))
                    item_list.append(filtered_item)
                # print(item)
            elif 'review' in sub_task['description']:
                # find all reviews from the user
                history_review = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        # DO NOT output your analysis process!??????????
        task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
                                Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
                                Please rank the more interested items more front in your rank list. The information of the above 20 candidate items is as follows: {item_list}.
                                Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
                                The correct output format:['item id1', 'item id2', 'item id3', ...]
        '''
        # task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        #                         Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        #                         Please rank the more interested items more front in your rank list. The information and reviews of the above 20 candidate items is as follows: {history_item_reviews}.
        #                         Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        #                         The correct output format:['item id1', 'item id2', 'item id3', ...]
        # '''
        # memory = self.memory(task_description)
        # print("memory:", memory)
        # prompt_len = num_tokens_from_string(task_description)
        # if prompt_len > 4096:
        #     print(f'prompt_len: {prompt_len}')
        result = self.reasoning(task_description)

        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print('Meta Output:',result)
                print("No list found.")
            print('Processed Output:',eval(result))
            time.sleep(2)
            return eval(result)
        except:
            print('format error')
            return ['']


if __name__ == "__main__":
    task_set = "yelp" # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="/AgentSocietyChallenge/data", device="auto", cache=False)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"/AgentSocietyChallenge/example/track2/{task_set}/tasks", groundtruth_dir=f"/AgentSocietyChallenge/example/track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    simulator.set_llm(InfinigenceLLM(api_key="sk-dapxrd44nc6qjgxk"))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=False, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}_unhack.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
