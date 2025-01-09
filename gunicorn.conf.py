# gunicorn.conf.py
workers = 1
timeout = 300
worker_class = 'sync'
bind = "0.0.0.0:10000"
preload_app = False  
max_requests = 1    
max_requests_jitter = 1 