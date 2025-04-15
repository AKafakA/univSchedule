import asyncio
import logging
import signal
import sys
from typing import Optional, Dict, Any, List

import psutil
import requests
import uvicorn
from aiohttp import ClientTimeout
from fastapi import FastAPI
import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logging.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logging.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


def get_http_request(query_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.get(query_url, headers=headers)
    return response


async def post_serving_request(url: str, data: Dict, client_timeout: ClientTimeout) -> Dict:
    """Send a POST request to the given URL with the given data.
    """
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to post request to {url}: {response.status}")


def set_ulimit(target_soft_limit=65535):
    if sys.platform.startswith('win'):
        return

    import resource
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type,
                               (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Failed to set soft limit to {target_soft_limit}: {e}")
            return
