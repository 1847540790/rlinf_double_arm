#!/bin/bash

# example: RLINF_NODE_RANK=0 RANK=0 bash ray_utils/start_ray.sh

# export RAY_worker_register_timeout_seconds=60
# export RAY_gcs_rpc_server_reconnect_timeout_s=60
# export RAY_heartbeat_timeout_milliseconds=120000
# export RAY_raylet_death_mark_period_seconds=120
# export RAY_task_lease_timeout_ms=60000


# Parameter check
if [ -z "$RLINF_NODE_RANK" ]; then
    echo "Error: RLINF_NODE_RANK environment variable not set!"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable not set!"
    exit 1
fi


# Configuration file path (modify according to actual needs)
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_PATH")
RAY_HEAD_IP_FILE=$REPO_PATH/ray_utils/ray_head_ip.txt
RAY_PORT=${RAY_PORT:-8400}  # Default port for Ray, can be modified if needed
PASSWORD=${PASSWORD:-"Noematrix_tmp123!"}

# Head node startup logic
if [ "$RANK" -eq 0 ]; then
    # Get local machine IP address (assumed to be intranet IP)
    # IP_ADDRESS=$(hostname -I | awk '{print $1}')
    IP_ADDRESS="14.103.157.86"
    # Start Ray head node
    echo "Starting Ray head node on rank 0, IP: $IP_ADDRESS"
    ray start --memory=10737418240 --head --port=$RAY_PORT   --node-ip-address=10.8.0.1   --dashboard-host=0.0.0.0   --dashboard-port=8265  --redis-password=$PASSWORD --verbose --include-dashboard=True

    # Write IP to file
    echo "$IP_ADDRESS" > $RAY_HEAD_IP_FILE
    echo "Head node IP written to $RAY_HEAD_IP_FILE"
else
    # Worker node startup logic
    echo "Waiting for head node IP file..."
    
    # Wait for file to appear (wait up to 360 seconds)
    for i in {1..360}; do
        if [ -f $RAY_HEAD_IP_FILE ]; then
            HEAD_ADDRESS=$(cat $RAY_HEAD_IP_FILE)
            if [ -n "$HEAD_ADDRESS" ]; then
                break
            fi
        fi
        sleep 1
    done
    
    if [ -z "$HEAD_ADDRESS" ]; then
        echo "Error: Could not get head node address from $RAY_HEAD_IP_FILE"
        exit 1
    fi
    
    echo "Starting Ray worker node connecting to head at $HEAD_ADDRESS"
    ray start --memory=10737418240 --address="$HEAD_ADDRESS:$RAY_PORT"
fi
