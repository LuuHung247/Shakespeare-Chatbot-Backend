# 🎭 Shakespeare Chatbot Backend

Backend API cho chatbot tạo thơ phong cách Shakespeare sử dụng FastAPI, LangChain và RAG.

## ✨ Tính năng

- 🤖 Hỗ trợ nhiều LLM: OpenAI, Anthropic Claude, hoặc Local models
- 📚 RAG (Retrieval-Augmented Generation) với Shakespeare's works
- 🎨 Nhiều style: Sonnet, Tragedy, Comedy, General
- 💬 Chat mode với Shakespearean personality
- 🔄 Linh hoạt: Dễ dàng switch giữa API và local model

## 📦 Cài đặt

### 1. Clone và setup environment

```bash
# Clone repo
git clone <your-repo>
cd shakespeare-chatbot-backend

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Cấu hình

```bash
# Copy .env.example sang .env
cp .env.example .env

# Edit .env và điền thông tin của bạn
nano .env  # hoặc dùng editor khác
```

````

## 🚀 Chạy server

```bash
# Development mode
python -m app.main
docker run -d -p 8000:8000 --name shakespeare shakespeare-app
# Hoặc dùng uvicorn trực tiếp
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

````

Server sẽ chạy tại: `http://localhost:8000`

## 📖 API Documentation

Sau khi chạy server, truy cập:

- Swagger UI: `http://localhost:8000/docs`
