import os, json
from dotenv import load_dotenv
from tavily import TavilyClient
from .schemas import RankingInput
from langchain_core.tools import tool, StructuredTool

load_dotenv()


@tool
def search_products(query: str, max_results: int = 5) -> str:
    """
    Search for real products matching the user query. Returns product names,
    prices, URLs, and review snippets from real e-commerce sites.
    Use when you need to find products matching specific requirements.
    """
    client  = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    results = client.search(query + ' price review buy 2025', max_results=max_results,
                            search_depth='advanced')
    if not results.get('results'):
        return 'No products found'
    out = []
    for r in results['results']:
        out.append(f"Product: {r['title']}\nURL: {r['url']}\nInfo: {r['content'][:300]}\n")
    return '\n'.join(out)


@tool
def search_product_reviews(product_name: str) -> str:
    """Search for expert and user reviews of a specific product."""
    client  = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    results = client.search(f'{product_name} review pros cons rating', max_results=3)
    out = []
    for r in results.get('results', []):
        out.append(f"Review Source: {r['title']}\n{r['content'][:400]}\n")
    return '\n'.join(out) or 'No reviews found'


def _rank_products(products: str, user_query: str, max_results: int = 3) -> str:
    return json.dumps({
        'ranking_complete': True,
        'message': f'Ranked top {max_results} products for: {user_query}',
        'products_evaluated': products[:200],
    })


rank_and_recommend = StructuredTool.from_function(
    func=_rank_products,
    name='rank_and_recommend',
    description=(
        'Rank products and generate final recommendation. '
        'Call this LAST after gathering product data and reviews. '
        'Provide all gathered data and the final ranking will be formatted.'
    ),
    args_schema=RankingInput,
)
