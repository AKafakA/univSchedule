from typing import Dict

from aiohttp import ClientTimeout

from api.instance_api_adapter.instance_adapter import Instance
from api.serve_utils import post_serving_request


class VLLMInstance(Instance):
    async def query_backend(self, request: Dict, max_tokens: int, client_timeout: ClientTimeout) -> Dict:
        """
        Query the VLLM instance with the given request.
        """
        # Implement the query logic for the VLLM instance here
        # For example, you can use an HTTP client to send a request to the VLLM instance
        # and return the response as a dictionary.
        prompt = request["prompt"]
        request_dict = {
            "prompt": prompt,
            "n": 1,
            "best_of": 1,
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": max(max_tokens, 1),
            "ignore_eos": True,
            "stream": False
        }
        return await post_serving_request(self._backend_url, request_dict, client_timeout)