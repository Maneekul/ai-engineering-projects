# ------------------------
# Running the Agent
# ------------------------

import sys
import os

# Ensure the parent directory of 'agentic_pattern' is in the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from process_pattern.reflection_pattern.reflection_agent import ReflectionAgent


# ------------------------------------------------------------------------------------------

# agent = ReflectionAgent()

# generation_system_prompt = "You are a Python programmer tasked with generating high quality Python code"
# reflection_system_prompt = "You are Andrej Karpathy, an experienced computer scientist"
# user_msg = "Generate a Python implementation of the Merge Sort algorithm"

# final_response = agent.run(
#     user_msg=user_msg,
#     generation_system_prompt=generation_system_prompt,
#     reflection_system_prompt=reflection_system_prompt,
#     n_steps=10
# )

# print("\nFinal Response:\n", final_response)