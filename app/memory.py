from typing import List, Dict

class ChatMemory:
    def __init__(self):
        self.history: List[Dict] = []

    def add(self, question: str, answer: str):
        self.history.append({"q": question, "a": answer})

    def get_formatted_history(self) -> str:
        if not self.history:
            return "No previous conversation."
        return "\n".join(f"Q: {h['q']}\nA: {h['a']}" for h in self.history)

    def clear(self):
        self.history.clear()

    def __len__(self):
        return len(self.history)