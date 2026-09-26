from fastapi.testclient import TestClient 
from app.main import app 
from app.schemas import BuildRequest 
import pytest 
from pydantic import ValidationError 
  
client = TestClient(app) 
  
def test_health(): 
    assert client.get("/health").json()["status"] == "ok" 
  
def test_request_validation(): 
    with pytest.raises(ValidationError): 
        BuildRequest(task="too short")     # below min_length 
  
@pytest.mark.live 
def test_build_runs_and_terminates(): 
    r = client.post("/build", json={"task": 
        "Write is_palindrome(s) ignoring case and spaces, with asserts."}) 
    assert r.status_code == 200 
    data = r.json() 
    assert data["rounds"] <= 12            # respected the hard cap 
    assert len(data["transcript"]) >= 2 