#MODEL_NAME=$1
#
#parallel-ssh -t 0 -h experiment/config/ollama_test/ollama_hosts "sudo apt update && sudo apt full-upgrade -y"
#parallel-ssh -t 0 -h experiment/config/ollama_test/ollama_hosts "sudo apt install -y python3-pip python3-venv ccache"
#parallel-ssh -t 0 -h experiment/config/ollama_test/ollama_hosts "pip3 install -U pip==25.0.1"
#parallel-ssh -t 0 -h experiment/config/ollama_test/ollama_hosts "curl -fsSL https://ollama.com/install.sh | sh"
#parallel-ssh -t 0 -h experiment/config/ollama_test/ollama_hosts "pip install ollama"
#parallel-ssh -t 0 -h experiment/config/ollama_test/ollama_hosts "ollama pull $MODEL_NAME"
#
#parallel-ssh -t 0 -h experiment/config/ollama_test/ollama_hosts "git clone https://github.com/AKafakA/univSchedule.git"
#parallel-ssh -i -t 0 -h experiment/config/ollama_test/ollama_hosts "cd univSchedule && pip install -r requirements.txt"
parallel-ssh -i -t 0 -h experiment/config/ollama_test/ollama_hosts "cd univSchedule && git reset --hard HEAD~2 && git pull"
parallel-ssh -i -t 0 -h experiment/config/ollama_test/ollama_hosts "cd univSchedule && export PYTHONPATH=. && python3 api/backend_entrypoint/ollama_entrypoint.py"


