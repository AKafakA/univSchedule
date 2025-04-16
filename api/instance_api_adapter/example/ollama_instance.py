from aiohttp import ClientTimeout

from api.instance_api_adapter.ollama_instance import OllamaInstance
import asyncio

ollama_instance = OllamaInstance(
    instance_id="ollama_instance",
    ip_address="localhost",
    port=8000,
    predict_model_root_path="",
    start_time_ms=0,
    lookback_steps=0,
    instance_type="2405",
    model_name="llama2:7b",
    model_maps={
        "llama2:7b": {
            "input_price": 0.0001,
            "output_price": 0.0002,
        }
    }
)


async def call():
    response = await ollama_instance.query_backend(
        request={"prompt": "Hello, how are you?"},
        max_tokens=100,
        client_timeout=ClientTimeout(total=10)
    )
    print(response)


asyncio.run(call())
