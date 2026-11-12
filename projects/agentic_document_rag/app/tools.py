from pathlib import Path
from projects.agentic_document_rag.app.retrieval import get_rag_engine
from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression containing numbers and operators.
    Use this only for arithmetic. Do not execute arbitrary Python code.
    """
    allowed = set("0123456789+-*/(). ")

    if not expression.strip() or any(character not in allowed for character in expression):
        return "Invalid arithmetic expression."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except (ArithmeticError, SyntaxError, TypeError, ValueError):
        return "Could not evaluate the arithmetic expression."

    return str(result)


@tool
def financial_report(company_name: str, revenue: float, expenses: float) -> str:
    """Generate a financial report for a company that calculates net income."""
    net_income = revenue - expenses
    report  = f"Financial Report for {company_name}:\n"
    report += f"Revenue: ${revenue:.2f}\n"
    report += f"Expenses: ${expenses:.2f}\n"
    report += f"Net Income: ${net_income:.2f}\n"
    return report


@tool
def check_available_files() -> list[str]:
    """List the exact filenames available for document retrieval."""
    return [path.name for path in get_rag_engine().available_files()]
    
    
@tool
def retrieve_document_context(query: str, file_name: str) -> str:
    """Retrieve evidence from a local document.

    Return evidence only. Do not answer the user's question.
    The agent will formulate the final answer using this evidence.
    """
    engine = get_rag_engine()

    requested_name = Path(file_name).name
    allowed_files = {
        path.name: path
        for path in engine.available_files()
    }

    file_path = allowed_files.get(requested_name)

    if file_path is None:
        available = ", ".join(sorted(allowed_files))
        return (
            f"Unknown file '{requested_name}'. "
            f"Available files: {available}"
        )

    documents = engine.retrieve(query, file_path)
    return engine.format_documents(documents)
