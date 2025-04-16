import argparse
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
import json

config_path = "experiment/config/ollama_test"
output_path = "experiment/config/ollama_test"


def generate_config(host_ip_address, host_backend_port, host_instance_type):
    config = {
        "ip_address": host_ip_address,
        "backend_port": host_backend_port,
        "instance_type": host_instance_type
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=config_path)
    parser.add_argument("--output_path", type=str, default=output_path)
    parser.add_argument("--host_username", type=str, default="asdwb")
    parser.add_argument("--backend_port", type=int, default=8000)
    args = parser.parse_args()

    manifest_path = os.path.join(args.config_path, "cl_manifest.xml")

    tree = ET.parse(manifest_path)
    # get root element
    nodes = {}
    root = tree.getroot()

    for child in root:
        if "node" in child.tag:
            node_info = {}
            node_name = child.get("client_id")
            nodes[node_name] = node_info
            for subchild in child:
                if "host" in subchild.tag:
                    ip_address = subchild.get("ipv4")
                    node_info["ip_adresses"] = ip_address
                if "services" in subchild.tag:
                    host_name = subchild[0].get("hostname")
                    node_info["hostname"] = host_name
                    node_info["node_type"] = host_name.split("-")[0]

    nodes = OrderedDict(sorted(nodes.items()))
    instance_config_files = os.path.join(args.config_path, "instance_type_config.json")
    instance_type_dict = json.load(open(instance_config_files))
    instances = {}
    frameworks = {}

    for node in nodes:
        node_info = nodes[node]
        instance_type = ""
        framework = ""
        for key, value in instance_type_dict.items():
            if value["node_type"] == node_info["node_type"]:
                instance_type = key
                framework = value["framework"]
                break
        if instance_type:
            instance_config = generate_config(node_info["ip_adresses"], args.backend_port, instance_type)
            host_name = node_info["hostname"]
            instances[host_name] = instance_config
            framework_host_address = frameworks.get(framework, [])
            framework_host_address.append(args.host_username + "@" + host_name)
            frameworks[framework] = framework_host_address

    output_instance_config_files = os.path.join(args.output_path, "instance_configs.json")
    with open(output_instance_config_files, "w+") as f:
        json.dump(instances, f, sort_keys=True, indent=4)

    for framework in frameworks:
        with open(os.path.join(args.output_path, framework+"_hosts"), "w+") as f:
            for host in frameworks[framework]:
                f.write(host + "\n")
