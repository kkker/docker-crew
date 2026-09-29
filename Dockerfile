
# 階段一：建構環境
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder
WORKDIR /app

# ⚡ 關鍵修正：透過環境變數強制 uv 在建立虛擬環境時放入 pip
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SEED=true

# 複製剛才在 Linux 容器裡生成好的兩個檔案
COPY pyproject.toml uv.lock ./
# 安裝相依套件（含 pip）
RUN uv sync --frozen --no-dev --no-install-project

# 再複製所有原始碼，確保 uv 完整同步專案
COPY . .
RUN uv sync --frozen --no-dev

## 用 Docker 來產生 uv.lock
# docker run --rm -v "$(pwd)":/app -w /app -e UV_PYTHON_PREFERENCE=managed ghcr.io/astral-sh/uv:python3.12-bookworm uv lock

# 階段二：最終執行環境
FROM python:3.12-slim-bookworm
WORKDIR /app

# 從 builder 複製組裝好的虛擬環境
COPY --from=builder /app/.venv /app/.venv

# 讓系統預設使用虛擬環境中的 Python
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# 提示：此時不需要在 Dockerfile 裡 COPY . .
# 因為 docker-compose.yml 會在運行時動態把你的程式碼掛載進來！
