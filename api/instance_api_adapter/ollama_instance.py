from typing import Dict
from api.instance_api_adapter.instance_adapter import Instance
from aiohttp import ClientTimeout

from api.serve_utils import post_serving_request


class OllamaInstance(Instance):
    def __init__(self, instance_id: str,
                 ip_address,
                 port,
                 predict_model_root_path,
                 start_time_ms,
                 lookback_steps,
                 instance_type,
                 model_name: str,
                 model_maps):
        super().__init__(instance_id, ip_address, port, predict_model_root_path,
                         start_time_ms, lookback_steps, instance_type, model_name, model_maps)

    async def query_backend(self, request: Dict, max_tokens: int, client_timeout: ClientTimeout) -> Dict:
        """Query the backend instance with the request.
        """
        # Implement the query logic for the Ollama instance here
        # For example, you can use an HTTP client to send a request to the Ollama instance
        # and return the response as a dictionary.
        prompt = request["prompt"]
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,  # Get the full response at once
            "options": {
                "num_keep": 24,
                "num_predict": max_tokens,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
            }
        }
        return await post_serving_request(self._backend_url, payload, client_timeout)
