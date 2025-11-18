#!/bin/bash
nohup uvicorn mainspacy:app  --host 0.0.0.0 --port 12012 &
