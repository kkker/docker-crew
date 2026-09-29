import os
from pypdf import PdfReader
from crewai import Crew, Agent, Task, LLM

from crewai.tools import tool
from typing import Union, Dict, Any
#
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ===== 關鍵修正 1：引入文本切塊工具 =====
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_URL = "http://192.168.1.225:11434"
EMBED_MODEL = "nomic-embed-text"

print("正在初始化本地 Ollama 向量模型...")
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_URL,
    model=EMBED_MODEL
)

vector_store = Chroma(
    collection_name="local_knowledge",
    embedding_function=embeddings
)
# ✨ 修正：先嘗試建立，並在每次重跑時徹底清空，避免舊資料交叉污染
try:
    vector_store.delete_collection() # 🔥 強制刪除舊的 collection
    print("🧹 已成功清空舊的 Chroma 向量資料庫快取。")
    # 重新初始化一個乾淨的
    vector_store = Chroma(collection_name="local_knowledge", embedding_function=embeddings)
except Exception:
    pass

# =========================================================================
# 二、 自動讀取、切塊並載入您的知識文件
# =========================================================================
#
# 修正 load_local_files() 函數（逐檔追蹤字數）
def load_local_files():
    # 改用字典來追蹤每個檔案讀到的字數
    files_content = {}
    
    # 1. 讀取 TXT 檔案
    txt_path = "knowledge/company_policy.txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            files_content["company_policy.txt"] = f.read()

    # 2. 讀取 MD 檔案
    md_path = "knowledge/product_spec.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            files_content["product_spec.md"] = f.read()

    # 3. 讀取 PDF 檔案
    pdf_path = "knowledge/product_spec.pdf"
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        pdf_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        files_content["product_spec.pdf"] = pdf_text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # ✨ 建議調整：將切塊縮小到 600 字，重疊 120 字，這能大幅提升精準度
        chunk_overlap=120,
        length_function=len
    )

    all_chunks = []
    # 🪲 精準檢查：印出每個檔案分別貢獻了幾個切塊
    for filename, text in files_content.items():
        chunks = text_splitter.split_text(text)
        print(f"📄 檔案 [{filename}] 成功切碎為 {len(chunks)} 個片段 (總字數: {len(text)})")
        all_chunks.extend(chunks)

    if all_chunks:
        vector_store.add_texts(texts=all_chunks)
        print(f"🚨 檢查：目前 Chroma 裡面總共有 {vector_store._collection.count()} 筆資料！")
        print("--- 所有本地知識庫向量化完成 ---")
    else:
        print("⚠️ 警告：未找到任何知識庫檔案。")


# 執行載入
load_local_files()

# =========================================================================
# 三、 自訂專屬的 RAG 檢索工具
# =========================================================================
@tool
def query_knowledge_tool(query: Union[str, Dict[str, Any]]) -> str:
    """當你需要查詢公司政策、產品規格、HR 規定或文檔中的詳細資料時，請使用此工具。"""
    
    if isinstance(query, dict):
        query_str = query.get("query") or query.get("description") or query.get("argument") or str(list(query.values()) if query else "")
    else:
        query_str = str(query)
        
    print(f"\n🔍 [工具內部接收] 框架實際傳入: '{query_str}'")

    # =========================================================================
    # ✨ 這裡是最容易擴充關鍵字的地方（未來只要在這邊自由新增一行即可）
    # =========================================================================
    KEYWORD_MAP = {
        # 如果使用者問了「左邊的詞」，就自動在 RAG 檢索時補上「右邊的陣列列表」
        "閘道器": ["閘道器", "Gateway", "規格", "硬體規格"],
        "設備":   ["設備", "硬體", "規格"],
        "電壓":   ["電壓", "輸入電壓", "電源", "Power"],
        "時脈":   ["時脈", "處理器", "CPU", "工作頻率"],
        "室外":   ["室外", "淋雨", "防水", "IP 等級", "Outdoor"],
        "報銷":   ["差旅報銷", "交通費用", "收據", "報帳"],
        "台鐵":   ["台鐵", "高鐵", "交通費用"],
        "請假":   ["請假", "假別", "特休", "加班"],
        "電壓":   ["電壓", "輸入電壓", "電源", "Power", "DC", "供電", "Power Input"],
    }
    ##
    # 例如，若您之後新增了有關 「薪水」 或 「年終」 的文件，而使用者可能會問「薪資單怎麼看」，您只需要在 KEYWORD_MAP 補上：
    # "薪水":   ["薪資", "薪水", "薪資單", "年終獎金", "Salary"],
    # "年終":   ["年終", "獎金", "績效考核"],

    # 1. 根據對照表自動擴充關鍵字
    extended_keywords = set() # 使用 set 避免一開始就重複
    
    for trigger_word, expand_list in KEYWORD_MAP.items():
        if trigger_word in query_str:
            extended_keywords.update(expand_list)

    # 如果沒有觸發任何關鍵字，就用 LLM 原本傳進來的大雜燴字串當作保底
    if not extended_keywords:
        extended_keywords = [query_str]
    else:
        extended_keywords = list(extended_keywords)

    # =========================================================================
    # 2. 物理強制檢索邏輯（保持不變）
    # =========================================================================
    all_results = []
    seen_contents = set()

    print(f"🧬 [雙語與專有名詞防禦加速] 擴充檢索詞列表: {extended_keywords}")
    for kw in extended_keywords:
        res = vector_store.similarity_search(kw, k=2)
        for doc in res:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                all_results.append(doc)

    print(f"DEBUG 🧭: 去重後最終合成了 {len(all_results)} 個片段。")
    for idx, doc in enumerate(all_results):
        clean_content = doc.page_content.replace('\n', ' ')
        print(f"  -> 片段 {idx+1}: {clean_content[:40]}...")

    if not all_results:
        return "在知識庫中找不到與該查詢相關的內容。"
    
    context = "\n\n".join([f"[片段 {i+1}]: {doc.page_content}" for i, doc in enumerate(all_results)])
    return f"【從內部知識庫檢索到的相關內容】：\n{context}"

# =========================================================================
# 四、 CrewAI 物件初始化與啟動
# =========================================================================
internal_ollama_llm = LLM(
    ## TODO, llama3.2 Not ready to test
    # model="ollama/llama3.2:latest",
    # llm_kwargs={
    #     "num_ctx": 4096,      # 擴大上下文到 8192，解決 500 噎到問題
    #     "num_predict": 512    # 設定單次最大生成長度（預設常被限制在 128 太短）
    # },
    # # options= {"num_ctx": 8192}, # Not work
    
    model="ollama/qwen2.5",
    base_url=OLLAMA_URL,
    temperature=0
)

hr_support_agent = Agent(
    role="高級智囊顧問",
    goal="精確根據公司內部知識庫回答提問，絕不胡思亂想。",
    backstory="你是一位專業、嚴謹的助理。你必須透過【query_knowledge_tool】工具來獲取正確答案。",
    tools=[query_knowledge_tool],
    llm=internal_ollama_llm,
    verbose=True
)

task1 = Task(
    description=(
        "有一位新進員工詢問了以下問題，請從公司的內部文件中找出答案：\n"
        # "1. 我們公司的最新差旅報銷政策與請假規定是什麼？\n"
        "1a. 台鐵交通費用要如何報銷？\n"
        "1b. 報銷期限是幾個工作天提交系統？\n"
        # "2. 這次新產品的技術規格亮點有哪些？\n"
        "2. 這台閘道器的處理器規格和時脈是多少？\n"
        # "3. 如果工廠設備只支援 Modbus RTU 且環境高達 70°C，這台設備能用嗎？要怎麼把資料傳上雲端？\n"
        "4. 這款閘道器可以放在室外淋雨環境使用嗎？\n"
        "5. 這台設備的額定輸入電壓（電壓範圍）是多少？\n\n"
        
        "請務必完全依據提供的知識庫文件內容來回答，切勿自己發明內容。"
    ),
    expected_output="一份結構完整、條列分明的繁體中文員工答詢指南，內容必須與內部文件完全一致。",
    agent=hr_support_agent
)

crew = Crew(
    agents=[hr_support_agent],
    tasks=[task1],
    verbose=True
)

print("--- CrewAI 啟動中 ---")
result = crew.kickoff()
print("\n=== 最終輸出結果 ===")
print(result)
