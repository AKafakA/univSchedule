import argparse
import asyncio
import ssl
import time
from typing import Any, Dict

from ollama import generate, AsyncClient
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from argparse import Namespace

from api.serve_utils import set_ulimit, serve_http

TIMEOUT_KEEP_ALIVE = 10  # seconds.
app = FastAPI()
asyncClient = AsyncClient()


async def get_response(model_name: str, prompt: str, options: Dict, start_time: float) -> Dict:
    response = {}
    generated_tokens = []
    time_stamps = []
    async for part in await asyncClient.generate(
            model=model_name,
            prompt=prompt,
            stream=True,
            options=options,
    ):
        time_stamp = time.time() * 1000 - start_time
        time_stamps.append(time_stamp)
        generated_tokens.append(part['response'])
    response["generated_response"] = " ".join(generated_tokens)
    response["ttft"] = time_stamps[0]
    if len(time_stamps) > 1:
        response["average_tbt"] = (time_stamps[1] - time_stamps[0]) / (len(time_stamps) - 1)
    else:
        response["average_tbt"] = 0.0
    response["output_length"] = len(generated_tokens)
    return response


@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    assert asyncClient is not None
    request_dict = await request.json()
    start_time = time.time() * 1000
    model_name = request_dict["model"]
    prompt = request_dict["prompt"]
    options = request_dict.get("options", {})
    response = await get_response(model_name, prompt, options, start_time)
    return JSONResponse(content=response)


async def run_server(args: Namespace,
                     **uvicorn_kwargs: Any) -> None:
    set_ulimit()
    global app, asyncClient
    app.root_path = args.root_path
    asyncClient = AsyncClient(
        host="http://localhost:" + str(args.backend_port),
    )

    shutdown_task = await serve_http(
        app,
        sock=None,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--backend_port", type=int, default=11434)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    args = parser.parse_args()
    asyncio.run(run_server(args))
