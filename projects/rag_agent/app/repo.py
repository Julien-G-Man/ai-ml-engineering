class Store:
    """ 
    Acts as our in-memory database layer.
    Persists messages and provides an interface to access stored messages. 
    Gets cleared one server stops
    """
        
    def __init__(self):
        self.past_messages: list[dict[str, str]] = []
        
    def save_conversation(self, user_message: str, ai_message: str):
        self.past_messages.append({"user": user_message, "ai": ai_message})
        
    def get_past_conversations(self) -> list[dict[str, str]]:
        return self.past_messages
    
    def clear_past_conversations(self):
        self.past_messages.clear()
        
    def clear_single_conversation(self, id: int):
        try:
           self.past_messages.remove(id - 1)
        except Exception as e:
            (f"Could not delete conversation: {e}")
    
    
store = Store()