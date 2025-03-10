
import json
import re
import requests
import math 
from colorama import Fore
from process_pattern.tools_pattern.tool import Tool

from process_pattern.reflection_pattern.reflection_agent import ReflectionAgent
from process_pattern.tools_pattern.tool import tool
from process_pattern.tools_pattern.tool_agent import ToolAgent

from process_pattern.planning_pattern.react_agent import ReactAgent


@tool
def sum_two_elements(a: int, b: int) -> int:
    """
    Computes the sum of two integers.

    Args:
        a (int): The first integer to be summed.
        b (int): The second integer to be summed.

    Returns:
        int: The sum of `a` and `b`.
    """
    return a + b


@tool
def multiply_two_elements(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a (int): The first integer to multiply.
        b (int): The second integer to multiply.

    Returns:
        int: The product of `a` and `b`.
    """
    return a * b

@tool
def compute_log(x: int) -> float | str:
    """
    Computes the logarithm of an integer `x` with an optional base.

    Args:
        x (int): The integer value for which the logarithm is computed. Must be greater than 0.

    Returns:
        float: The logarithm of `x` to the specified `base`.
    """
    if x <= 0:
        return "Logarithm is undefined for values less than or equal to 0."
    
    return math.log(x)


available_tools = {
    "sum_two_elements": sum_two_elements,
    "multiply_two_elements": multiply_two_elements,
    "compute_log": compute_log
}

agent = ReactAgent(tools=[sum_two_elements, multiply_two_elements, compute_log])

agent.run(user_msg="I want to calculate the sum of 1234 and 5678 and multiply the result by 5. Then, I want to take the logarithm of this result")
