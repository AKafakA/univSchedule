TARGET_HOST="asdwb@c240g5-110209.wisc.cloudlab.us"
RESTART_BACKEND=true
RUN_EXP=true

parallel-ssh -t 0 --host $TARGET_HOST "pkill -f api_server"

if [ "$RESTART_BACKEND" = "true" ]; then
  nohup sh experiment/scripts/run_ollama_backend.sh > /dev/null 2>&1 &
fi

if [ "$RUN_EXP" = "true" ]; then
  # Run the experiment
fi




