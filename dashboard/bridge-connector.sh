#!/bin/bash
# Connects ci-dashboard to bridge network for Plex access
# Retries every 10s for 5 min, then every 5 min thereafter

CONTAINER="ci-dashboard"
NETWORK="bridge"
FAST_INTERVAL=10
FAST_DURATION=300  # 5 minutes
SLOW_INTERVAL=300  # 5 minutes

start_time=$(date +%s)

while true; do
    # Check if container is running
    if docker inspect "$CONTAINER" &>/dev/null; then
        # Check if already connected to bridge
        if ! docker inspect "$CONTAINER" --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' | grep -q "$NETWORK"; then
            if docker network connect "$NETWORK" "$CONTAINER" 2>/dev/null; then
                echo "$(date): Connected $CONTAINER to $NETWORK"
            fi
        fi
    fi

    # Determine sleep interval
    elapsed=$(($(date +%s) - start_time))
    if [ $elapsed -lt $FAST_DURATION ]; then
        sleep $FAST_INTERVAL
    else
        sleep $SLOW_INTERVAL
    fi
done
