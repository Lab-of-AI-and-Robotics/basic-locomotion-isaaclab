#!/bin/bash

# Preload runtime libs needed on some ARM64 conda setups where torch/cv2 fail
# with "cannot allocate memory in static TLS block".

if [ -z "${GO2_LIBGOMP_PRELOAD_READY:-}" ]; then
    _preload_libs=()

    if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/lib/libgomp.so.1" ]; then
        _preload_libs+=("$CONDA_PREFIX/lib/libgomp.so.1")
    elif [ -f "/usr/lib/aarch64-linux-gnu/libgomp.so.1" ]; then
        _preload_libs+=("/usr/lib/aarch64-linux-gnu/libgomp.so.1")
    fi

    if [ -f "/lib/aarch64-linux-gnu/libGLdispatch.so.0" ]; then
        _preload_libs+=("/lib/aarch64-linux-gnu/libGLdispatch.so.0")
    elif [ -f "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0" ]; then
        _preload_libs+=("/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0")
    fi

    if [ ${#_preload_libs[@]} -gt 0 ]; then
        if [ -n "${LD_PRELOAD:-}" ]; then
            export LD_PRELOAD="${_preload_libs[*]} $LD_PRELOAD"
        else
            export LD_PRELOAD="${_preload_libs[*]}"
        fi
    fi

    export GO2_LIBGOMP_PRELOAD_READY=1
    unset _preload_libs
fi
