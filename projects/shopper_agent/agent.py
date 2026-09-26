import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from shopper_agent.tools import search_products, search_product_reviews, rank_and_recommend

load_dotenv()


SHOPPER_INSTRUCTIONS = '''You are an expert personal shopping assistant.
When given a shopping query:
1. Search for matching products with search_products
2. Get reviews for the top 2-3 candidates using search_product_reviews
3. Call rank_and_recommend with your analysis
4. Write a Final Answer with: top pick, runner-up, and why each was chosen

Always mention: product name, approximate price, key pros, key cons.
Be specific, users want actionable recommendations, not vague suggestions.'''


class AgentRunner:
    def __init__(self, agent, verbose=False):
        self.agent = agent
        self.verbose = verbose

    def invoke(self, input_dict):
        user_text = input_dict.get('input') if isinstance(input_dict, dict) else str(input_dict)
        result = self.agent.invoke({'messages': [HumanMessage(content=user_text)]})
        messages = result.get('messages', [])
        output = messages[-1].content if messages else ''
        steps_taken = sum(1 for msg in messages if getattr(msg, 'tool_call_id', None) is not None)
        return {'messages': messages, 'output': output, 'intermediate_steps': [None] * steps_taken, 'raw': result}


def build_shopper_agent(verbose=False):
    tools  = [search_products, search_product_reviews, rank_and_recommend]
    llm    = ChatOpenAI(model='gpt-4o-mini', temperature=0.1, api_key=os.getenv('OPENAI_API_KEY'))
    agent  = create_agent(llm, tools, system_prompt=SHOPPER_INSTRUCTIONS, debug=verbose)
    return AgentRunner(agent, verbose=verbose)

