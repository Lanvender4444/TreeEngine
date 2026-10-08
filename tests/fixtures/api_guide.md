---
title: Orbit SDK Developer Guide
version: 2.3
---

Orbit SDK lets you upload telemetry from devices to the Orbit cloud.

Installation
============

Install the package with pip:

```bash
pip install orbit-sdk==2.3.1
# this line is a comment, not a heading
```

Authentication
--------------

Create a client with an API key. Keys start with the prefix `ork_` and expire after 90 days.

```python
from orbit import Client

client = Client(api_key="ork_xxx")
```

## Uploading Data

### Batch upload

Use `client.upload_batch(records)` to send up to 500 records per request.
Requests larger than 5 MB are rejected with error `E413_PAYLOAD_TOO_LARGE`.

### Streaming

The `client.stream()` context manager keeps a WebSocket open and flushes every 2 seconds.

- reconnects automatically with exponential backoff
- maximum backoff is 30 seconds
- set `max_retries=0` to disable reconnects

## Rate Limits

| Plan | Requests per minute |
| --- | --- |
| Free | 60 |
| Pro | 1200 |

When the limit is exceeded the API returns HTTP 429 with a `Retry-After` header.

## Error Handling

> All errors inherit from `OrbitError`.

The function `retry_with_jitter()` is recommended for transient failures such as `E503_UNAVAILABLE`.
