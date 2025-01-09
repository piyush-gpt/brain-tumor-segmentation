workers = 1
timeout = 300
worker_class = 'sync'
bind = "0.0.0.0:10000"
preload_app = True  # Preload the app to avoid reloading
max_requests = 10   # Restart workers periodically
max_requests_jitter = 5  # Add randomness to worker restarts
