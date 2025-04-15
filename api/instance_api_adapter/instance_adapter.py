import time
from abc import ABC, abstractmethod
from typing import Dict

import aiohttp
from aiohttp import ClientTimeout
import torch
from lightning import LightningModule
from torch import Tensor

MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 10000  # in milliseconds


def convert_request_trace_to_tensors(request: Dict) -> Tensor:
    """Convert the request trace to a tensor.
    """
    return torch.tensor([request["timestamp"],
                         request["latency"],
                         request["input_length"],
                         request["output_length"]], dtype=torch.float32)


def convert_request_to_predictable_request(request: Dict, instance_id: int) -> Dict:
    current_time = time.time() * 1000
    return {
        "request_id": int(request["request_id"]),
        "timestamp": int(current_time),
        "latency": -1,
        "input_length": int(request["prompt_len"]),
        "output_length": int(request["expected_response_len"][instance_id])
    }


def get_max_tokens_with_budget(request: Dict) -> int:
    """Adjust the expected response length with the budget.
    """
    if "budget" in request:
        budget = request["budget"]
        leftover_budget = budget - request["input_price"] * request["prompt_len"]
        if leftover_budget < 0:
            return -1
        max_expected_length = min(MAX_TOKENS, leftover_budget // request["output_price"])
    else:
        max_expected_length = MAX_TOKENS
    return max_expected_length


class Instance(ABC):
    def __init__(self,
                 instance_id,
                 ip_address,
                 port,
                 predict_model_root_path,
                 start_time_ms,
                 lookback_steps,
                 instance_type,
                 model_name: str,
                 model_maps: Dict,
                 ) -> None:
        self.instance_type = instance_type
        self.instance_id = instance_id
        self._ip_address = ip_address
        self._port = port
        self._backend_url = f"http://{ip_address}:{port}/generate"
        if predict_model_root_path:
            predict_model_path = predict_model_root_path + "/" + str(instance_type) + "/"
            self._predict_model = LightningModule.load_from_checkpoint(predict_model_path)
        else:
            self._predict_model = None
        self._submitted_requests = {}
        self._start_time_ms = start_time_ms
        self._lookback_steps = lookback_steps
        self._model_name = model_name
        self._input_price = model_maps[model_name]["input_price"]
        self._output_price = model_maps[model_name]["output_price"]

    def prepare_input_for_predictor(self, predicted_request) -> Tensor:
        first_unfinished_request_index = next((i for i, x in enumerate(self._submitted_requests) if x['latency'] > 0),
                                              len(self._submitted_requests))
        first_index = max(first_unfinished_request_index - self._lookback_steps, 0)
        lookback_requests = []
        for i in range(first_index, len(self._submitted_requests)):
            enqueued_request = self._submitted_requests[i]
            lookback_requests.append(convert_request_trace_to_tensors(enqueued_request))
        lookback_requests.append(convert_request_trace_to_tensors(
            convert_request_to_predictable_request(predicted_request)))
        return torch.stack(lookback_requests)

    @abstractmethod
    async def query_backend(self, request: Dict, max_tokens: int, client_timeout: ClientTimeout) -> Dict:
        """Query the instance with the given request.
        """
        pass

    async def predict(self, request: Dict) -> Dict:
        """Query the predictor with the given request.
        """
        assert self._predict_model is not None, "Predictor model is not loaded"
        predicted_request = dict(request)
        predicted_length = predicted_request["expected_response_len"][self.instance_id]
        max_tokens_with_budget = get_max_tokens_with_budget(predicted_request)

        if max_tokens_with_budget < 0:
            return {"error": "Budget exceeded"}

        predicted_request["expected_response_len"][self.instance_id] = \
            min(max_tokens_with_budget, predicted_length)

        predicted_metrics = self._predict_model(
            self.prepare_input_for_predictor(predicted_request)
        )
        predicted_metrics = predicted_metrics.squeeze().tolist()
        predicted_request["predicted_latency"] = predicted_metrics[0]
        predicted_request["instance_id"] = self.instance_id
        predicted_request["price"] = self._input_price * request["prompt_len"] + \
                                     self._output_price * request["expected_response_len"][self.instance_id]
        return predicted_request

    async def generate(self, request: Dict) -> Dict:
        """
        Generate the request with the given request.
        """
        max_tokens = get_max_tokens_with_budget(request)
        predictable_request = convert_request_to_predictable_request(request, self.instance_type)

        timeout_in_ms = request.get("timeout", DEFAULT_TIMEOUT)
        client_timeout = aiohttp.ClientTimeout(total=timeout_in_ms / 1000)
        response = await self.query_backend(request, max_tokens, client_timeout)
        latency = predictable_request["timestamp"] - self._start_time_ms

        predictable_request["latency"] = latency
        predictable_request["response"] = response
        predictable_request["output_length"] = response["output_length"]
        self._submitted_requests[predictable_request["request_id"]] = predictable_request
        return response
