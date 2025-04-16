parallel-ssh -t 0 --host $HOST "pkill -f ollama_entrypoint"
sleep 10
parallel-ssh -t 0 -h experiment/config/ollama_test/ollama_hosts "cd univSchedule && export PYTHONPATH=. && python3 api/backend_entrypoint/ollama_entrypoint.py"