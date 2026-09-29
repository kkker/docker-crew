# 確保有安裝 langchain-ollama 才能成功匯入
from langchain_ollama import OllamaEmbeddings 

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://192.168.1.225:11434"
)

# 測試轉換
vector = embeddings.embed_query("哈囉，這是一段測試")
print(vector[:5]) # 印出前五個維度數值

# Ok after "pip install -U langchain-core langchain-ollama"