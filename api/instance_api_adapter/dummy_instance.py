import random
from typing import Dict

from aiohttp import ClientTimeout

from api.instance_api_adapter.instance_adapter import Instance

dummy_corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "In the middle of difficulty lies opportunity.",
    "The only thing we have to fear is fear itself.",
    "The greatest glory in living lies not in never falling, but in rising every time we fall.",
    "The purpose of our lives is to be happy.",
    "Life is what happens when you're busy making other plans.",
    "Get busy living or get busy dying.",
]


class DummyInstance(Instance):
    """
    Dummy instance API that does nothing.
    """

    async def query_backend(self, request: Dict, max_tokens: int, client_timeout: ClientTimeout) -> Dict:
        """
        Query the instance with the given request.
        response: {
            "request_id": int,
            "generated_response": str,
            "TTFT": float,
            "generated_length": float,
        }
        """
        # randomly select a response from the dummy corpus
        response = random.choice(dummy_corpus)
        length = len(response.split())
        return {
            "request_id": request["request_id"],
            "generated_response": response,
            "TTFT": random.uniform(0.1, 0.5),  # simulate some processing time
            "generated_length": length,
        }
