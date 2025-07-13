#!/bin/bash
gunicorn myapp.app:app --bind 0.0.0.0:$PORT
