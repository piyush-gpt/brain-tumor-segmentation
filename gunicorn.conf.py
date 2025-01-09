# gunicorn.conf.py
workers = 1
timeout = 300  # 5 minutes to allow for model loading
worker_class = 'sync'
bind = "0.0.0.0:10000"