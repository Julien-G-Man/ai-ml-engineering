from crewai.tools import tool 
  
# Curated, conservative reference data. In production this would be a vetted 
# clinical knowledge API.
SYMPTOM_REFERENCE = { 
    "chest pain": { 
        "common_context": ["muscle strain", "acid reflux", "anxiety"], 
        "red_flags": ["crushing/pressure pain", "pain to arm or jaw", 
                      "shortness of breath", "cold sweat", "nausea"], 
        "urgency": "HIGH", 
    }, 
    "headache": { 
        "common_context": ["tension", "dehydration", "eye strain", "migraine"], 
        "red_flags": ["sudden worst-ever headache", "fever + stiff neck", 
                      "confusion", "weakness/numbness on one side"], 
        "urgency": "MODERATE", 
    }, 
    "abdominal pain": { 
        "common_context": ["indigestion", "gas", "muscle strain", "mild infection"], 
        "red_flags": ["rigid/board-like abdomen", "pain with vomiting blood", 
                      "severe lower-right pain", "fainting"], 
        "urgency": "MODERATE", 
    }, 
} 
  
@tool("Symptom Reference Lookup") 
def symptom_reference(symptom: str) -> str: 
    """Return vetted reference context and recognised emergency red-flags for a 
    symptom. Use this to ground the research brief. If the symptom is not in the 
    reference, say so explicitly rather than inventing information.""" 
    key = symptom.lower().strip() 
    for name, data in SYMPTOM_REFERENCE.items(): 
        if name in key or key in name: 
            return str({"symptom": name, **data}) 
    return (f"No curated reference exists for '{symptom}'. State this gap clearly " 
            "and recommend professional assessment.") 