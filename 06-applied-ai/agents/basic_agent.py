import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    api_key=os.getenv('OPENAI_API_KEY'),
)


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together. Use when you need to compute a sum."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers. Use when you need to compute a product."""
    return a * b


@tool
def get_current_year() -> int:
    """Get the current year. Use when the calculation involves the current year."""
    from datetime import datetime
    return datetime.now().year

prompt = 'What is 37 multiplied by 48, then add 2025 minus the current year?'

tools = [add_numbers, multiply_numbers, get_current_year]
agent = create_agent(llm, tools, debug=False)

result = agent.invoke({'messages': [HumanMessage(content=prompt)]})
print('\nFINAL ANSWER:', result['messages'][-1].content)
