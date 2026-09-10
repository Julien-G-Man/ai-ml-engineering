from engine import RAGEngine, Agent

agent = None
rag_engine = None


def get_agent() -> Agent:
    global agent
    if agent is None:
        agent = Agent()
    return agent


def get_rag_engine() -> RAGEngine:
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


agent = Agent()
rag_engine = RAGEngine()


def main():
    query = "I want to build my own Jarvis, like Jarvis from IronMan"

    documents, sources = rag_engine.retrieve(query, top_k=3)
    context_prompt = agent.build_prompt_with_context(query, documents)

    response = agent.generate_response(context_prompt, sources)
    print(response)


if __name__ == "__main__":
    print(main())
