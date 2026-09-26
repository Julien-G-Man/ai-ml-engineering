from fastapi.testclient import TestClient 
from app.main import app 
from app.schemas import TriageRequest 
import pytest 
from pydantic import ValidationError 
  
client = TestClient(app) 
  
def test_health(): 
    assert client.get("/health").json()["status"] == "ok" 
  
def test_request_validation_rejects_short_input(): 
    with pytest.raises(ValidationError): 
        TriageRequest(complaint="hi")   # below min_length 
  
def test_reference_tool_known_symptom(): 
    from app.tools import symptom_reference 
    out = symptom_reference.run("chest pain") 
    assert "red_flags" in out and "HIGH" in out 
  
def test_reference_tool_unknown_symptom(): 
    from app.tools import symptom_reference 
    out = symptom_reference.run("left elbow tingle") 
    assert "No curated reference" in out 
  
# Mark the full crew run as 'live' (it calls the LLM) so it is opt-in: 
@pytest.mark.live 
def test_full_triage_runs(): 
    r = client.post("/triage", json={"complaint": 
        "Mild headache for two days, worse in the evening, drinking little water."}) 
    assert r.status_code == 200 
    assert len(r.json()["briefing"]) > 50 