class Store:
    def __init__(self):
        self.past_messages: list[dict[str, str]] = []
        
    def save_message(self, user_message: str, ai_message: str):
        self.past_messages.append({"user": user_message, "ai": ai_message})
        
    def get_past_messages(self) -> list[dict[str, str]]:
        return self.past_messages
    
    
store = Store()