import json
import requests
from colorama import Fore


import sys
sys.path.append(r'C:\Users\kantaphong\Desktop\Work Station\Work_inet\agentic_pattern\process_pattern')

# Now try the import
from process_pattern.utils.completions import build_prompt_structure


# from process_pattern.utils.completions import build_prompt_structure
# from process_pattern.utils.completions import completions_create
# from process_pattern.utils.completions import FixedFirstChatHistory
# from process_pattern.utils.completions import update_chat_history
# from process_pattern.utils.logging import fancy_step_tracker


# from utils.completions import ChatHistory

# ------------------------
# AI Model API Function
# ------------------------
def ai_llm(text):
    url = "https://ai-api.manageai.co.th/llm-model-04/v1/chat/completions"

    payload = json.dumps({
        "model": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ],
        "temperature": 0,
        "max_tokens": 400
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Basic bWFuYWdlYWkyMDI0Ok1hbmFnZUFJQDIwMjQ=',
        'Cookie': 'Path=/'
    }

    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"


# ------------------------
# ReflectionAgent Class
# ------------------------
class ReflectionAgent:
    def __init__(self):
        self.model = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"

    def _request_completion(self, text, log_title="", log_color=Fore.WHITE):
        output = ai_llm(text)
        print(log_color, f"\n\n{log_title}\n\n", output)
        return output

    def generate(self, text):
        return self._request_completion(text, log_title="GENERATION", log_color=Fore.BLUE)

    def reflect(self, text):
        return self._request_completion(text, log_title="REFLECTION", log_color=Fore.GREEN)

    def run(self, user_msg, generation_system_prompt="", reflection_system_prompt="", n_steps=10):
        BASE_GENERATION_SYSTEM_PROMPT = """
        Your task is to Generate the best content possible for the user's request.
        If the user provides critique, respond with a revised version of your previous attempt.
        You must always output the revised content.
        """

        BASE_REFLECTION_SYSTEM_PROMPT = """
        You are tasked with generating critique and recommendations to the user's generated content.
        If the user content has something wrong or something to be improved, output a list of recommendations
        and critiques. If the user content is ok and there's nothing to change, output this: <OK>
        """

        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        generation_history = f"{generation_system_prompt}\nUser: {user_msg}"
        reflection_history = reflection_system_prompt

        for step in range(n_steps):
            print(f"\nStep {step + 1}/{n_steps}:")

            generation = self.generate(generation_history)
            reflection = self.reflect(generation)

            if "<OK>" in reflection:
                print(Fore.RED, "\n\nStopping reflection loop...\n\n")
                break

            generation_history = f"{generation_system_prompt}\nUser: {user_msg}\nAssistant: {generation}\nCritique: {reflection}"

        return generation



