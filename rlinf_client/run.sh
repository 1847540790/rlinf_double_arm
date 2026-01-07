#!/bin/bash
_load_env(){
    # set frpc.toml localPort and remotePort according to the env file
    
    # Load environment variables from .env file if it exists
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    # Set default values if not provided
    LOCAL_PORT=${RAY_CLIENT_PORT:-28101}
    REMOTE_PORT=${RAY_CLIENT_PORT:-28101}
    
    # Update frpc.toml with the port values using sed
    sed -i "s/^localPort = .*/localPort = ${LOCAL_PORT}/" frpc.toml
    sed -i "s/^remotePort = .*/remotePort = ${REMOTE_PORT}/" frpc.toml
    
    echo "Updated frpc.toml: localPort=${LOCAL_PORT}, remotePort=${REMOTE_PORT}"
    
}

start(){
   
    docker compose --env-file .env up -d
}

stop(){
    docker compose down
}

restart(){
    stop
    start
}

status(){
    docker compose ps
}


_load_env
case $1 in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac