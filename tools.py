import time

import requests
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from loguru import logger

items = list()


class ItemModel(BaseModel):
    item_name: str = Field(..., description="The name of the item")


class RequestModel(BaseModel):
    status_code: int = Field(..., description="The status code of the request")
    url: str = Field("https://http.cat", description="The URL to call")


class AdderModel(BaseModel):
    a: int = Field(..., description="The number")
    b: int = Field(..., description="The number")


@tool("add", args_schema=AdderModel)
def add(a: int, b: int) -> int:
    """adds two numbers and returns the sum"""
    logger.info(f"Adding {a} and {b}")
    return a + b


@tool("add_item", args_schema=ItemModel)
def add_item(item_name: str) -> dict:
    """add an item at a specific time to the list of items and return the item"""
    t = time.asctime()
    logger.info(f"Adding {item_name} at {t}")
    item = {"item_name": item_name, "time": t}
    items.append(item)
    print(f"Items: {items}")
    return item


@tool("get_catcode", args_schema=RequestModel)
def get_catcode(status_code: int, url: str = "https://http.cat") -> int:
    """
    get the status code of a request to a URL
    """
    response = requests.get(url + "/" + str(status_code))
    logger.info(f"Calling {url}")
    return response.status_code
