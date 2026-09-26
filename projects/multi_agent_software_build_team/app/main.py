from fastapi import FastAPI, HTTPException 
from app.schemas import BuildRequest, BuildResponse 
from app.build_team import run_build 

app = FastAPI(title="AutoGen Build Team", version="1.0.0") 

  
@app.get("/health") 
def health(): 
    return {"status": "ok", "service": "build-team"} 
  
@app.post("/build", response_model=BuildResponse) 
def build(req: BuildRequest): 
    try: 
        out = run_build(req.task) 
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Build team failed: {e}")
    return BuildResponse(**out)