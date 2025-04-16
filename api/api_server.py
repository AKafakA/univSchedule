import argparse
import json
import random
import ssl
import time
from argparse import Namespace
from enum import Enum

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from api.instance_api_adapter.instance_adapter import Instance
from schedule.scheduler import Scheduler
from serve_utils import *
import resource
import logging


class ExecutionMode(Enum):
    TRAIN = "train"
    SERVING = "serving"


lock = asyncio.Lock()
TIMEOUT_KEEP_ALIVE = 10  # seconds.
app = FastAPI()
instances = []
num_requests = 0
scheduler = Scheduler({})
start_time_in_ms = 0
logging.basicConfig(level=logging.INFO,
                    filemode='a+',
                    filename='experiment_output/logs/scheduler_output.log')
logger = logging.getLogger(__name__)
execution_mode = ExecutionMode.TRAIN


@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    """Generate completion for the request with profiling.
        request: {
            "request_id": int,
            "prompt": str,
            "prompt_len": int,

            # optional during serving used for scheduling
            "expected_response_len": Dict[str, int],
            "relevance_score": [float],
            "budget": float,
            "timeout": int,
        }


        => predicted_request: request + {
            "request_id": int,
            "predicted_latency": float,
            "instance_id": int,
            "price": int
        }


        response: {
            "request_id": int,
            "generated_response": str,
            "time_stamp": float,
            "instance_type": str,
            "TTFT": float,
            "average_tbt": float,
            "output_length": int,
        }
    """
    assert len(instances) > 0
    request_dict = await request.json()
    arrived_at = time.time() * 1000 - start_time_in_ms
    _ = request_dict.pop("stream", False)
    global scheduler
    request_id = request_dict["request_id"]
    if execution_mode == ExecutionMode.SERVING:
        assert scheduler is not None
        enhanced_requests = []
        for instance in instances:
            predicted_latency = instance.predict(request_dict)
            if "error" in predicted_latency:
                return JSONResponse(predicted_latency)
            else:
                enhanced_requests.append(instance.predict(request_dict))
        selected_instance_index = scheduler.schedule(enhanced_requests)

    # if training, random send the request to all instances and collect the instance data
    elif execution_mode == ExecutionMode.TRAIN:
        selected_instance_index = random.randint(0, len(instances) - 1)
    else:
        raise ValueError(f"Unknown execution mode: {execution_mode}")
    response = await instances[selected_instance_index].generate(request_dict)
    response["time_stamp"] = arrived_at
    response["request_id"] = request_id
    response["instance_type"] = instances[selected_instance_index].instance_type
    finished_at = time.time() * 1000 - start_time_in_ms
    response["latency"] = finished_at - arrived_at
    return JSONResponse(response)


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def init_app(
        args: Namespace,
        instances_list: Optional[List[Instance]] = None,
) -> FastAPI:
    app = build_app(args)
    global instances, start_time_in_ms, scheduler, execution_mode
    instance_config_path = args.instance_config_path
    scheduler_config_path = args.scheduler_config_path
    model_path = args.prediction_model_path
    start_time_in_ms = time.time() * 1000
    instance_dict = json.load(open(instance_config_path))
    scheduler_dict = json.load(open(scheduler_config_path))
    instance_type_dict = json.load(open(args.instance_type_config_path))
    scheduler = Scheduler(scheduler_dict)
    execution_mode = ExecutionMode(args.phase)

    if instances_list is not None:
        instances.extend(instances_list)
    else:
        for key, value in instance_dict.items():
            instance_type = value["instance_type"]
            instance = Instance(key, value["ip_address"], value["backend_port"],
                                model_path, start_time_in_ms, args.lookback_steps, value["instance_type"],
                                value["model_name"],
                                instance_type_dict[instance_type])
            instances.append(instance)
    assert len(instances) > 0
    return app


async def run_server(args: Namespace,
                     **uvicorn_kwargs: Any) -> None:
    app = await init_app(args)
    assert len(instances) > 0

    if args.debugging_logs:
        logger.setLevel(logging.DEBUG)

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        workers=args.workers,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8200)
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
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--instance_config_path",
                        type=str, default="experiment/config/instance_config.json")
    parser.add_argument("--scheduler_config_path",
                        type=str, default="experiment/config/scheduler_config.json")
    parser.add_argument("--instance_type_config_path",
                        type=str, default="experiment/config/instance_type_config.json")
    parser.add_argument("--prediction_model_path", type=str,
                        default="",
                        help="prediction models only used for serving")
    parser.add_argument("--debugging_logs", type=bool, default=False)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--lookback_steps", type=int, default=1)
    args = parser.parse_args()
    # in case the limited by the number of files
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    asyncio.run(run_server(args))
