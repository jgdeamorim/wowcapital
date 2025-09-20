import os
import requests
import pymongo
import redis

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://mongo:27017/wowcapital')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://qdrant:6333')
API_URL = os.getenv('API_URL', 'http://localhost:8080')

print('== Redis ==')
redis_client = redis.Redis.from_url(REDIS_URL)
print('PING:', redis_client.ping())

print('\n== Mongo ==')
mongo_client = pymongo.MongoClient(MONGO_URI)
print('PING:', mongo_client.admin.command('ping'))

print('\n== Qdrant ==')
try:
    health = requests.get(f"{QDRANT_URL}/healthz", timeout=5)
    health.raise_for_status()
    print('HEALTH:', health.text.strip())
except Exception as exc:
    print('Qdrant check error:', exc)

print('\n== API ==')
try:
    resp = requests.get(f"{API_URL}/docs", timeout=5)
    print('/docs status:', resp.status_code)
except Exception as exc:
    print('API check error:', exc)
