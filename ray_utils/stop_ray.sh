#!/bin/bash

# Script to stop Ray and clean up worker processes
# Usage: bash ray_utils/stop_ray.sh

echo "=== Stopping Ray processes ==="

# Stop Ray if it's running
if command -v ray &> /dev/null; then
    echo "Stopping Ray cluster..."
    ray stop --force 2>/dev/null || true
else
    echo "Ray command not found, trying to kill Ray processes directly..."
fi

# Kill any remaining Ray processes
echo "Killing remaining Ray processes..."
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
pkill -9 -f "dashboard" 2>/dev/null || true

# Clean up Ray runtime files (optional, uncomment if needed)
# echo "Cleaning Ray runtime files..."
# rm -rf /tmp/ray* 2>/dev/null || true
# rm -rf /tmp/session_* 2>/dev/null || true

#!/bin/bash
echo "🧹 清理 Ray 残留进程和状态..."
pkill -9 -f ray
rm -rf /tmp/ray /tmp/ray_session_* ~/.ray/ /dev/shm/plasma* /dev/shm/ray*
echo "✅ 清理完成，请重新启动 Ray"

echo "=== Ray cleanup complete ==="
echo "You can now restart Ray with: bash ray_utils/start_ray.sh"

