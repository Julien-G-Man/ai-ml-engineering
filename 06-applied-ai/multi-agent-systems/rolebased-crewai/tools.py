from crewai.tools import tool 
  
# A curated, conservative reference table (in production: a vetted clinical API) 
SYMPTOM_REFERENCE = { 
    "chest pain": { 
        "common_context": ["muscle strain", "acid reflux", "anxiety"], 
        "red_flags": ["crushing pain", "pain spreading to arm/jaw", 
                      "shortness of breath", "sweating"], 
        "urgent": True, 
    }, 
    "headache": { 
        "common_context": ["tension", "dehydration", "screen strain", "migraine"], 
        "red_flags": ["worst headache of life", "fever with stiff neck", 
                      "confusion", "weakness on one side"], 
        "urgent": False, 
    }, 
}


  
@tool("Symptom Reference Lookup") 
def symptom_reference(symptom: str) -> str: 
    """Look up general reference context and recognised emergency red-flags 
    for a symptom. Use to ground the research brief in vetted information. 
    Returns common context, red-flags, and whether the symptom is high-urgency.""" 
    key = symptom.lower().strip() 
    for name, data in SYMPTOM_REFERENCE.items(): 
        if name in key or key in name: 
            return str({name: data}) 
    return f"No curated reference for '{symptom}'. Note this gap explicitly." 