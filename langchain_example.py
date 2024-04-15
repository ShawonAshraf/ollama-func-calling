# https://github.com/cfahlgren1/natural-functions/blob/main/natural-functions-demo.ipynb

from typing import Optional, Dict
import re
from langchain_core.utils.function_calling import convert_to_openai_function


from loguru import logger

import ollama
import json

from tools import add_item, get_catcode, add
from typing import Any

from tqdm.auto import tqdm



def parse_function_call(input_str: str) -> Optional[Dict[str, Any]]:
    """
    Parses a text string to find and extract a function call.
    The function call is expected to be in the format:
    <functioncall> {"name": "<function_name>", "arguments": "<arguments_json_string>"}

    Args:
        input_str (str): The text containing the function call.

    Returns:
        Optional[Dict[str, any]]: A dictionary with 'name' and 'arguments' if a function call is found,
                                  otherwise None.
    """
    # Regex pattern to extract 'name' and 'arguments'
    pattern = r'"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*?)\''

    # Search with regex
    match = re.search(pattern, input_str)
    if match:
        try:
            name = match.group(1)
            arguments_str = match.group(2)

            # Parse the arguments JSON
            arguments = json.loads(arguments_str)

            return {"name": name, "arguments": arguments}
        except json.JSONDecodeError:
            # If JSON parsing fails, return None
            return None
    return None


def call_llm(messages: list):
    response = ollama.chat(
        model='calebfahlgren/natural-functions', messages=messages)
    message = (response['message']['content']) # type: ignore
    return message


def get_function(content_str: str, functions: list):
    logger.info(f"User input: {content_str}")
    
    try_count = 0
    
    
    SYSTEM_PROMPT = f"""
    You are a helpful assistant with access to these functions -
    {json.dumps(functions, indent=4)}
    """
    
    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': content_str},
    ]
    message = call_llm(messages)

    function_call = parse_function_call(message)
    while not function_call:
        try_count += 1
        logger.error("Failed to parse function call")
        logger.info(f"Retrying -> Count: {try_count}")

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': content_str},
        ]
        
        message = call_llm(messages)

        messages.append({'role': 'assistant', 'content': message})
        # parse out function call name and args into json
        function_call = parse_function_call(message)
    
    return function_call


if __name__ == "__main__":
    tools = [add_item, get_catcode, add]
    tool_names = [t.get_name() for t in tools]
    
    
    functions = [convert_to_openai_function(t) for t in tools]
    
    
    # content_str = "send code 102"
    contents = [
        "send code 102",
        "add the number 999 with -999",
        "add apple to the list",
    ]
    

    for idx, content_str in tqdm(enumerate(contents), total=len(contents)):
        function_call = get_function(content_str, functions)
        idx = tool_names.index(function_call['name'])
        tool = tools[idx]
        args = function_call.get('arguments')
        result = tool.run(args)
        logger.info(f"Result: {result}")

             
        
        

        
        


