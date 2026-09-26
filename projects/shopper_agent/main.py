import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from shopper_agent.agent import build_shopper_agent
from shopper_agent.schemas import ShopRequest, ShopResponse

shop_app = FastAPI(title='Smart Shopper Agent', version='1.0.0')
shop_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

_shopper = None


@shop_app.on_event('startup')
async def startup():
    global _shopper;
    _shopper = build_shopper_agent(verbose=False)


@shop_app.get('/health')
async def health():
    return {'status':'healthy'}


@shop_app.post('/recommend', response_model=ShopResponse)
async def recommend(req: ShopRequest):
    if not _shopper: raise HTTPException(503,'Agent not ready')
    t0 = time.time()
    result = _shopper.invoke({'input': f'Find me the best {req.query}. Top {req.top_n} picks.'})
    return ShopResponse(
        query=req.query, recommendations=result['output'],
        steps_taken=len(result.get('intermediate_steps',[])),
        latency_ms=round((time.time()-t0)*1000, 1),
    )
