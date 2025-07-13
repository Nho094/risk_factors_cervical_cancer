#!/bin/bash
cd myapp
gunicorn app:app --bind 0.0.0.0:$PORT

