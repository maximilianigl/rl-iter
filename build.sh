#!/usr/bin/env bash
if hash nvidia-docker 2>/dev/null; then
    nvidia-docker build -t iterimage .
else
    docker build -t iterimage .
fi
