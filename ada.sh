#!/usr/bin/bash

srun --mem=20000 --gres=gpu:2 --time=1:00:00 --constraint=rtx_2080  --pty --preserve-env python ada_endpoint.py 