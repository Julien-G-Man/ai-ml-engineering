from rag_agent.app.engine import RAGEngine, Agent

agent = Agent()
rag_engine = RAGEngine()


def run_agent():
    query = "I want to build my own Jarvis, like Jarvis from IronMan"

    documents, sources = rag_engine.retrieve(query, top_k=3)
    context_prompt = agent.build_prompt_with_context(query, documents)

    response = agent.generate_response(context_prompt, sources)
    print(response)


if __name__ == "__main__":
    print(run_agent())
