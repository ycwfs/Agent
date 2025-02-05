import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryBase
from websocietysimulator.agent.modules.tooluse_modules import ToolUseBase 
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

class SummaryTool(ToolUseBase):
    def __init__(self, llm):
        super().__init__(llm=llm)
    
    def __call__(self, reviews: str = None, item: str = None, item_id: str = None):
#         prompt = f'''Analyze the data provided in {reviews} containing customer reviews for product {item_id}. 
# Your task is to generate a concise summary that includes:
# Overall Sentiment: Predominant positive/negative/neutral ratio (quantify if possible)
# Key Strengths:
# List 3-5 most frequently mentioned positive attributes (e.g. durability, ease of use)
# Include specific praise quotes when notable
# Common Criticisms:
# List 3-5 most repeated complaints
# Note any severity patterns (e.g. "20% mentioned delivery issues")
# Emerging Patterns:
# Unusual praise/criticism worth highlighting
# Contradictory opinions (e.g. "some found X intuitive while others struggled")
# Recommendation Insight:
# Typical user profile this product suits best
# Who should avoid it based on reviews
# Structure the summary with clear headings and bullet points. Maintain neutral tone while preserving review nuances. Highlight any statistically significant findings if detectable from the data.
# '''
#         prompt = f'''Analyze {reviews} data. 50-word summary for item {item_id}:
# Sentiment: [%+/-]
# Top 3 Pros (with frequency%)
# Top 2 Cons (severity🔴/🟡)
# Key Stat (e.g. '60% praised X')
# User Fit: [Best for.../Avoid if...]
# Use ➕/➖ symbols. No markdown. Strict 50 words.
# '''
        if reviews:
            print('reviews:',reviews)
            prompt = f'''Analyze {reviews} data. Strict 50-word summary for item {item_id}:
    Sentiment ratio (positive/neutral/negative %)
    Top 3 praised features with frequency
    Top 2 criticisms with severity (high/moderate)
    Key statistical insight from data
    Ideal user profile vs avoidance scenario
    Use only text and numbers. 
    '''
        if item:
            print('item:',item)
            prompt = f'''Summarize the key details of the following item info in a concise format. Include the most relevant information such as id, category, ratings, features, and any notable attributes. Keep the summary brief and to the point.
    Example Format:
    Item_id: [Item id]
    Category: [Primary Category]
    Rating: [Average Rating] (if available)
    Key Features: [List 2-3 main features or attributes]
    Notable Details: [Any unique or important information]

    Item info:
    {item}
'''
        messages = [{"role": "user", "content": prompt}]
        summary = self.llm(messages=messages,temperature=0.1)
        print('summary:',summary)
        return summary

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
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)
        #self.memory = RecMemory(llm=self.llm)
        self.summary = SummaryTool(llm=self.llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user = ''
        item_list = []
        summary_item_reviews = []
        history_reviewss = []
        # loc = self.task['loc']
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
                    item_id = item['item_id']
                    item_statictis = str(item)

                    item_reviews = self.interaction_tool.get_reviews(item_id=self.task['candidate_list'][n_bus])
                    item_reviews = str(item_reviews)
                    input_tokens = num_tokens_from_string(item_reviews)
                    if input_tokens > 12000:
                        encoding = tiktoken.get_encoding("cl100k_base")
                        item_reviews = encoding.decode(encoding.encode(item_reviews)[:12000]) + '}'

                    # summary each review, use item_statictis or item_id
                    summary_item_reviews.append(self.summary(item=item_statictis) + '\n review summary: ' + self.summary(reviews = item_reviews,item_id = item_id))
                    # summary_item_reviews.append(item_id + '\n review summary: ' + self.summary(reviews = item_reviews,item_id = item_id))


            elif 'review' in sub_task['description']:
                # find all reviews from the user
                history_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                for history_review in history_reviews:
                    item_id = history_review['item_id']
                    item_statictis = str(self.interaction_tool.get_item(item_id=item_id))
                    history_review = str(history_review)
                    input_tokens = num_tokens_from_string(history_review)
                    if input_tokens > 12000:
                        encoding = tiktoken.get_encoding("cl100k_base")
                        history_review = encoding.decode(encoding.encode(history_review)[:12000])
                    history_reviewss.append(self.summary(item=item_statictis) + '\n review summary: ' + self.summary(reviews = history_review,item_id = item_id))
                    # history_reviewss.append(item_id + '\n review summary: ' + self.summary(reviews = history_review,item_id = item_id))

            else:
                pass
        # DO NOT output your analysis process!??????????
        # task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        #                         Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        #                         Please rank the more interested items more front in your rank list. The information of the above 20 candidate items is as follows: {item_list}.
        #                         Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        #                         The correct output format:['item id1', 'item id2', 'item id3', ...]
        # '''
        task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_reviewss}. 
    Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
    Please rank the more interested items more front in your rank list. The information and reviews of the above 20 candidate items is as follows: {summary_item_reviews}.
    Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format,
    DO NOT introduce any other item ids!
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
        print('----------------------------------------------------------------------------')
        print('task_description:',task_description)
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
    
    #simulator.set_llm(InfinigenceLLM(api_key="sk-dapxrd44nc6qjgxk"))
    simulator.set_llm(OpenAILLM(api_key="sk-ppL21f5be930e1e868145d1a8d891975ae07c9b6c4aNYc2c"))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}_unhack_summary_prompt.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
