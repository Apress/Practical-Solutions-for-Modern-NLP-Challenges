# Rule-based response function (simplified example)
def respond_to_user(user_message: str) -> str:
    text = user_message.lower()
    if "visa requirements" in text:
        return "For international travel, you'll need a valid passport and possibly a visa depending on your destination."
    elif "office hours" in text or "open hours" in text:
        return "Our office is open from 9 AM to 5 PM, Monday to Friday."
    else:
        return "I'm sorry, I can only answer questions about visa requirements or office hours at the moment."
