curl http://localhost:8004/v1/chat/completions -H "Content-Type: application/json"  -d '{
    "model": "/mnt/models/DeepSeek-V4-Flash-BF16/",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "stream": false
  }'