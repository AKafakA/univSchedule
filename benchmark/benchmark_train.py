# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import functools

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import aiohttp
import argparse
import asyncio
import json
import os
import random
import time
import numpy as np
import sys
from enum import Enum
from transformers import AutoTokenizer
from typing import List
import resource

resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

num_finished_requests = 0
server_num_requests = {}


def get_wait_time(qps: float, distribution: str, burstiness: float = 1.0) -> float:
    mean_time_between_requests = 1.0 / qps
    if distribution == "uniform":
        return mean_time_between_requests
    elif distribution == "gamma":
        # variance = (coefficient_variation * mean_time_between_requests) ** 2
        # shape = mean_time_between_requests ** 2 / variance
        # scale = variance / mean_time_between_requests
        assert burstiness > 0, (
            f"A positive burstiness factor is expected, but given {burstiness}.")
        theta = 1.0 / (qps * burstiness)
        return np.random.gamma(shape=burstiness, scale=theta)
    else:
        return np.random.exponential(mean_time_between_requests)


def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return


async def async_request_gen(generator, qps: float, distribution="uniform", burstiness: float = 0.0):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                await asyncio.sleep(get_wait_time(qps, distribution, burstiness))
        except StopIteration:
            return


class GenerationBackend(str, Enum):
    univSched = "univSched"


async def query_model_univSched(prompt, verbose, ip_ports, with_request_id=True, max_request_len=0):
    prompt, prompt_len, request_id = prompt

    # Evenly dispatch request to the given api servers.
    global server_num_requests
    server_id = min(server_num_requests, key=server_num_requests.get)
    server_num_requests[server_id] += 1
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    global num_finished_requests

    async with aiohttp.ClientSession(timeout=timeout) as session:
        request_dict = {
            "prompt": prompt,
            "prompt_len": prompt_len,
        }
        if with_request_id:
            request_dict["request_id"] = request_id

        if verbose:
            print('Querying model')
        try:
            async with session.post(f'http://{ip_ports[server_id]}/generate_benchmark', json=request_dict) as resp:
                if verbose:
                    print('Done')
                output = await resp.json()
                return prompt, output
        except aiohttp.ClientError as e:
            print(f"Connect to {ip_ports[server_id]} failed with: {str(e)}")
            sys.exit(1)


def load_prompts(prompt_file):
    with open(prompt_file) as f:
        prompts = [json.loads(l) for l in f.readlines()]
    return prompts


def get_tok_id_lens(tokenizer, batch):
    tokenized = tokenizer.batch_encode_plus(batch)
    lens = [len(s) for s in tokenized['input_ids']]
    return lens


def get_token_ids(input_str, tokenizer):
    t = tokenizer(input_str)
    return t['input_ids']


async def run_with_sampling(
        backend: GenerationBackend,
        tokenizer,
        prompts: List[tuple],
        verbose: bool,
        ip_ports: List[int],
        distribution: str,
        qps: float,
        burstiness: float,
        max_request_len: int,
        shuffle: bool = True,
        record_context: bool = False,
):
    if backend == GenerationBackend.univSched:
        query_model = query_model_univSched
    else:
        raise ValueError(f'unknown backend {backend}')

    global server_num_requests
    num_servers = len(ip_ports)
    for server_id in range(num_servers):
        server_num_requests[server_id] = 0

    if distribution == "burst":
        qps = float('inf')

    if shuffle:
        random.shuffle(prompts)

    async_prompts = async_request_gen(
        iter(prompts), qps=qps, distribution=distribution, burstiness=burstiness)

    tasks = []
    async for prompt in async_prompts:
        tasks.append(asyncio.create_task(query_model(
            prompt,
            verbose=verbose,
            ip_ports=ip_ports,
            max_request_len=max_request_len,
        )))
    sampled_prompts = []
    sampled_responses = []
    training_data = []
    queries = await asyncio.gather(*tasks)

    for prompt, output in queries:
        if 'generated_text' in output:
            sampled_prompts.append(prompt)
            sampled_responses.append(output['generated_text'])
            training_data.append({
                "request_id": output['request_id'],
                "time_stamp": output['time_stamp'],
                "instance_type": output['instance_type'],
                "latency": output['latency'],
            })
            if 'ttft' in output:
                training_data[-1]['ttft'] = output['ttft']

            if record_context:
                training_data[-1]['prompt'] = prompt
                training_data[-1]['response'] = output['generated_text']

    sampled_responses_length = get_tok_id_lens(tokenizer, sampled_responses)
    sampled_prompts_length = get_tok_id_lens(tokenizer, sampled_prompts)
    for i in range(len(sampled_prompts)):
        training_data[i]['input_length'] = sampled_prompts_length[i]
        training_data[i]['output_length'] = sampled_responses_length[i]
    return training_data


# Sample requests from the dataset with long input and output.
# this is used for arxiv and code generation tasks and the will need to be converted to sharegpt format for later run
def sample_long_request(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        max_seqlen: int,
        task: str
):
    prompts = []
    prompt_lens = []
    with open(dataset_path) as f:
        for id_, row in enumerate(f):
            data = json.loads(row)
            if task == "arxiv":
                prompt = " ".join(data["article_text"])
            elif task == "code":
                prompt = data["input"]
            prompt_token_ids = tokenizer(prompt).input_ids
            if max_seqlen > len(prompt_token_ids) > 0:
                prompts.append(prompt)
                prompt_lens.append(len(prompt_token_ids))

    sampled_ids = [random.randint(0, len(prompts) - 1) for _ in range(num_requests)]
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    return sampled_prompts, sampled_prompt_lens


def sample_sharegpt_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        max_seqlen: int,
):
    # Load the dataset.
    prompts = []
    prompt_lens = []
    dataset = []
    if dataset_path.endswith('.jsonl'):
        with open(dataset_path) as f:
            for line in f:
                dataset.append(json.loads(line))
    elif dataset_path.endswith('.json'):
        with open(dataset_path) as f:
            dataset = json.load(f)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    for data in dataset:
        prompt = data["conversations"][0]["value"]
        prompt_token_ids = tokenizer(prompt).input_ids
        if max_seqlen > len(prompt_token_ids) > 0:
            prompts.append(prompt)
            prompt_lens.append(len(prompt_token_ids))

    sampled_ids = [random.randint(0, len(prompts) - 1) for _ in range(num_requests)]
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    return sampled_prompts, sampled_prompt_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument('--trust_remote_code',
                        action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--backend', type=GenerationBackend,
                        choices=[e.name for e in GenerationBackend], default='univSched')
    parser.add_argument('--train_data_path', type=str, default='/experiment_output/train_data.json')
    parser.add_argument('--ip_ports', nargs='+', required=True, help='List of ip:port')
    parser.add_argument('--num_sampled_requests_per_runs', type=int, default=10)
    parser.add_argument('--max_request_len', type=int, default=8152)
    parser.add_argument(
        '--distribution', choices=["uniform", "gamma", "exponential"], default="gamma")
    parser.add_argument('--qps', type=float, default=4.0)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--num_train_data', type=int, default=10001)
    parser.add_argument('--burstiness', type=float, default=1.0)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_type', type=str, choices=['sharegpt', 'code', 'arxiv'])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_path', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.gen_random_prompts:
        assert args.num_sampled_requests is not None

    backend = GenerationBackend[args.backend]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)

    if args.dataset_type == "sharegpt":
        prompts, prompt_lens = sample_sharegpt_requests(args.dataset_path,
                                                        args.num_sampled_requests_per_runs,
                                                        tokenizer, args.max_request_len)
    elif args.dataset_type == "arxiv":
        prompts, prompt_lens = sample_long_request(args.dataset_path,
                                                   args.num_sampled_requests_per_runs,
                                                   tokenizer,
                                                   task="arxiv",
                                                   max_seqlen=args.max_request_len)
    elif args.dataset_type == "code":
        prompts, prompt_lens = sample_long_request(args.dataset_path,
                                                   args.num_sampled_requests_per_runs,
                                                   tokenizer,
                                                   task="code",
                                                   max_seqlen=args.max_request_len)
    else:
        raise ValueError("unknown dataset type")

    packed_prompts = [(prompt, prompt_len, i) for i, (prompt, prompt_len) in enumerate(zip(prompts, prompt_lens))]
    num_runs = args.num_runs
    training_data = []
    # if num_runs == 1, it usually means we are running the benchmark to collect the new data trace for training models
    record_context = num_runs == 1
    assert args.num_train_data > 0, "num_train_data should be greater than 0"
    assert num_runs != 0, "num_runs should not be 0"
    if num_runs > 1:
        # use the first prompt as the prompt for the training data
        for i in range(num_runs):
            print(f"Running benchmark {i + 1} / {num_runs}")
            training_data += asyncio.run(run_with_sampling(
                backend=backend,
                tokenizer=tokenizer,
                prompts=packed_prompts,
                verbose=args.verbose,
                ip_ports=args.ip_ports,
                distribution=args.distribution,
                qps=args.qps,
                burstiness=args.burstiness,
                max_request_len=args.max_request_len,
                shuffle=True,
                record_context=record_context,
            ))
            if len(training_data) > args.num_train_data:
                break
            print(f"Finished runs {i + 1} / {num_runs}")
    else:
        sampled_num = -1 * len(packed_prompts) / num_runs
        while len(training_data) < args.num_train_data:
            random_sampled_prompts = random.sample(packed_prompts, int(sampled_num))
            training_data += asyncio.run(run_with_sampling(
                backend=backend,
                tokenizer=tokenizer,
                prompts=random_sampled_prompts,
                verbose=args.verbose,
                ip_ports=args.ip_ports,
                distribution=args.distribution,
                qps=args.qps,
                burstiness=args.burstiness,
                max_request_len=args.max_request_len,
                shuffle=True,
                record_context=record_context,
            ))
            print(f"Finished runs {len(training_data)} / {args.num_train_data}")

    training_data = training_data[:args.num_train_data]
    print(f"Finished all runs, total {len(training_data)} requests.")
    with open(args.train_data_path, 'w') as f:
        json.dump(training_data, f)


if __name__ == '__main__':
    main()
