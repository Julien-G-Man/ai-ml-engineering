from pathlib import Path
from pypdf import PdfReader
from transformers import pipeline

document = Path(__file__).resolve().parent.parent / "files/pdf" / "Masters-AI-Engineering.pdf"
if not document.exists():
    raise FileNotFoundError(f"PDF not found: {document}")

reader = PdfReader(str(document))

document_text = ""
for page in reader.pages:
    document_text += page.extract_text() or ""

qa_pipeline = pipeline(
    task="text-generation",
    model="bigscience/bloomz-560m",
)

question = "What are the keypoints?"
prompt = (
    "Answer the question using only the document context.\n\n"
    f"Context:\n{document_text[:4000]}\n\n"
    f"Question: {question}\nAnswer:"
)
result = qa_pipeline(prompt, max_new_tokens=80, do_sample=False)[0]["generated_text"]

answer = result.split("Answer:", 1)[-1].strip()
print(f"Answer: {answer}")


"""
qa_pipeline = pipeline(
    task="document-question-answering",
    model="distilbert-base-cased-distilled-squad"
)

question = "What are the modules in the program"
result = qa_pipeline(
    question=question,
    context=document_text,
)
print(f"Answer: {result.get("answer")}")
"""
