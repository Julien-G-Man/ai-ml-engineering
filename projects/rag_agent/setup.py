from setuptools import setup

setup(
    name="rag_agent",
    version='0.0.1',
    description='AI agent with RAG and tools',
    author='Julien Glory Manana',
    author_email='juliengmanana@gmail.com',
    packages=[''],
    install_requires=[
        'numpy', 
        'openai',
        'pandas', 
        'pinecone',
        'anthropic',
        'python-dotenv'
    ]
)
