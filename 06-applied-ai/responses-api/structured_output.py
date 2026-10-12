import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys_prompt = """You are a Spanish vocabulary tutor. Grade the student's quiz answers. 
Grade the quiz with two points per correct answer"""


class Mistake(BaseModel):
    word:           str = Field(description="The spanish word that was incorrect")
    student_answer: str = Field(description="What the student wrote")
    correct_answer: str = Field(description="The correct translation")
    
class DetailedQuizResult(BaseModel):
    score:    int  = Field(description="Number of correct answers out of the total")
    passed:   bool = Field(description="True if score >= 7")
    feedback: str  = Field(description="Encouraging message with specific tips for improvement")
    mistakes: list[Mistake] = Field(description="List of incorrect answers")
    
    
def generate_response(input: str):
    response = client.responses.parse(
        model="gpt-5.4-mini",
        instructions=sys_prompt,
        input=input,
        text_format=DetailedQuizResult
    )
    return response.output_parsed


def main(input):
    result: DetailedQuizResult = generate_response(input)
    print(type(result))
    print(f"Score: {result.score}/10")
    print(f"Passed: {result.passed}")
    for mistake in result.mistakes:
        print(f"{mistake.word}: '{mistake.student_answer}' -> '{mistake.correct_answer}' ")
    print(f"Feedback: {result.feedback}")
    

user_answers = """
1. casa = house
2. perro = dog
3. gato = cat
4. libro = book
5. agua = water
"""

if __name__ == "__main__":
    main(user_answers)
    
    
