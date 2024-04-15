import json
from typing import List, Sequence

import nest_asyncio
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.ollama import Ollama
from openai.types.chat import ChatCompletionMessageToolCall

nest_asyncio.apply()


from tools import add, add_item, call_ollama, items

add_tool = FunctionTool.from_defaults(fn=add)
ollam_tool = FunctionTool.from_defaults(fn=call_ollama)
item_tool = FunctionTool.from_defaults(fn=add_item)


class FuncCallerAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: Ollama = Ollama(model="calebfahlgren/natural-functions", temperature=0.0),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._llm = llm
        self._chat_history = chat_history

    def _call_function(self, tool_call: ChatCompletionMessageToolCall) -> ChatMessage:
        id_ = tool_call.id
        function_call = tool_call.function
        tool = self._tools[function_call.name]
        output = tool(**json.loads(function_call.arguments))

        return ChatMessage(
            name=function_call.name,  # type: ignore
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call.name,
            },
        )

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))  # type: ignore
        tools = [tool.metadata.to_openai_tool() for _, tool in self._tools.items()]

        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        tool_calls = additional_kwargs.get("tool_calls", None)
        # parallel function calling is now supported
        if tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content  # type: ignore


if __name__ == "__main__":
    agent = FuncCallerAgent(tools=[add_tool, ollam_tool, item_tool])
    # r = agent.chat("Hi, What is the sum of 999 and -999?")
    # print(r)
    # agent.reset()
    # r = agent.chat("Hi, send a get request to http://localhost:11434")

    # these work
    # r = agent.chat("what is the status code returned from the function call?")
    # print(r)
    # agent.reset()

    # r = agent.chat("what is the status code returned by the tool?")
    # print(r)
    # agent.reset()

    # # gives a generic answer
    # r = agent.chat("what is the response status code?")
    # print(r)
    # agent.reset()

    r = agent.chat("add item apple at  time 12 am")
    print(r)
    print(items)
