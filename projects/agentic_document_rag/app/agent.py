from __future__ import annotations

import os
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
from app.prompts import AGENT_SYSTEM_PROMPT
from app.tools import (calculate, financial_report, 
    check_available_files, retrieve_document_context)
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not configured")

BASE_DIR = Path(__file__).resolve().parents[2]
FILES_DIR = (BASE_DIR / "files").resolve()

CSV_PATH = FILES_DIR / "datasets" / "advertising_and_sales_clean.csv"
PDF_PATH = FILES_DIR / "pdf" / "Masters-AI-Engineering.pdf"


class Agent:
    """Tool-using agent with document-grounded RAG capability."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY,
        )
        self.tools = [retrieve_document_context, calculate,
            check_available_files, financial_report, 
        ]

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=AGENT_SYSTEM_PROMPT
        )
        

    def chat(self, message: str) -> str:
        if not message.strip():
            raise ValueError("Message cannot be empty")

        result = self.agent.invoke({"messages": [
            {"role": "user", "content": message}]
        })
        
        messages = result.get("messages", [])
        if not messages:
            raise RuntimeError("Agent returned no messages")

        content: Any = messages[-1].content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "\n".join(part for part in text_parts if part)
        
        return str(content)



def display_tool_info() -> None:
    for registered_tool in (
        retrieve_document_context,
        check_available_files,
        financial_report,
        calculate,
    ):
        print(f"\nName: {registered_tool.name}")
        print(f"Description: {registered_tool.description}")
        print(f"Arguments: {registered_tool.args}")


   

    
if __name__ == "__main__":
    agent = Agent()
    q = "Tell me more about this program. Does it prepare me for embedded systems?"
    print("Agent reponse: ", agent.chat(q))

    print("\n\nDirect RAG Output:")
    print(retrieve_document_context.invoke({"query": q, "file_name": str(PDF_PATH.name),}))