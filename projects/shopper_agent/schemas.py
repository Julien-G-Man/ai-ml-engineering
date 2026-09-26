from pydantic import BaseModel, Field

class ShopRequest(BaseModel):
    query:  str   = Field(..., min_length=5, examples=['wireless headphones under $100'])
    top_n:  int   = Field(3, ge=1, le=5)


class ShopResponse(BaseModel):
    query:       str
    recommendations: str
    steps_taken: int
    latency_ms:  float


class RankingInput(BaseModel):
    products:    str = Field(..., description='JSON string listing products with names, prices, pros, cons')
    user_query:  str = Field(..., description='The original user shopping request')
    max_results: int = Field(3, description='Number of top recommendations to return')
