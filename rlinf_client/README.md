# Update .env and frpc.toml first
avoid public name and port conflicts
.env:
```
RAY_CLIENT_PORT=28101
```

frpc.toml:
```
name = "ray-client-1"
```

# Usage
```
"Usage: ./run.sh {start|stop|restart|status}"
```

# Start
```
chmod +x run.sh
./run.sh start
```