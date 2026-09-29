import os
from pypdf import PdfReader
from typing import List
from crewai import Crew, Agent, Task, LLM
from crewai.tools import tool
from typing import Union, Dict, Any

# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma



# 確保之前的 OLLAMA_URL 與 query_knowledge_tool 工具依然在全域可調用
OLLAMA_URL = "http://192.168.1.225:11434"
EMBED_MODEL = "nomic-embed-text"

# ===== 關鍵修正 1：引入文本切塊工具 =====
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
def load_local_files_old():
    raw_texts = []

    # # 1. 讀取 TXT 檔案
    # txt_path = "knowledge/company_policy.txt"
    # if os.path.exists(txt_path):
    #     with open(txt_path, "r", encoding="utf-8") as f:
    #         raw_texts.append(f.read())
    #     print(f"成功加載 TXT 知識庫：{txt_path}")

    # 2. 讀取 MD 檔案
    md_path = "knowledge/product_spec.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            raw_texts.append(f.read())
        print(f"成功加載 MD 知識庫：{md_path}")

    # 3. 讀取 PDF 檔案
    pdf_path = "knowledge/product_spec.pdf"
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        pdf_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        raw_texts.append(pdf_text)
        print(f"成功加載 PDF 知識庫：{pdf_path}")

    if raw_texts:
        # 將所有文章拼成一個大文本
        combined_text = "\n\n".join(raw_texts)
        
        # ===== 關鍵修正 2：設定切塊規則 =====
        # 每個區塊最多 1000 個字，重疊 200 個字以防前後文意中斷
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # 將長文章切碎成一個 List 陣列
        final_chunks = text_splitter.split_text(combined_text)
        print(f"📑 文章太長，已成功切碎為 {len(final_chunks)} 個文字片段。")
        
        # 將切碎的片段丟進 Chroma，Ollama 就能一口一口吃下去，不會再噎到（500 錯誤）
        vector_store.add_texts(texts=final_chunks)
        print("--- 所有本地知識庫向量化完成 ---")
    else:
        print("⚠️ 警告：未找到任何知識庫檔案。")

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
# 三、 終極修正：將型態改為 Union 阻斷 Pydantic 報錯，並自動拆解 CrewAI 傳參
# =========================================================================

@tool
def query_knowledge_tool_old(query: Union[str, Dict[str, Any]]) -> str:
    """當你需要查詢公司政策、產品規格、HR 規定或文檔中的詳細資料時，請使用此工具。
    輸入參數應該是你想查詢的關鍵字或核心問題。"""
    
    # 【核心防禦】如果 CrewAI 還是丟了 {'description': '...'} 或類似字典進來，在此精準還原為字串
    if isinstance(query, dict):
        query_str = (
            query.get("query") or 
            query.get("description") or 
            query.get("argument") or 
            str(list(query.values())[0] if query else "")
        )
    else:
        query_str = str(query)
        
    # 如果被 CrewAI 轉成了無效的空字串，給予一個預設導向，防止 RAG 報錯
    if not query_str.strip():
        query_str = "查詢公司內部知識庫"
        
    print(f"\n🔍 [工具觸發] 正在為您向內網 Ollama 檢索關鍵字: '{query_str}'...")
    
    try:
        # 尋找最相關的 3 個文字片段
        results = vector_store.similarity_search(query_str, k=3)
        if not results:
            return "在知識庫中找不到與該查詢相關的內容。"
        
        context = "\n\n".join([f"[片段 {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])
        print("✅ [工具成功] 檢索完成！")
        return f"【從內部知識庫檢索到的相關內容】：\n{context}"
        
    except Exception as e:
        error_msg = f"內網 Ollama 檢索伺服器暫時沒有回應或發生錯誤: {str(e)}"
        print(f"❌ [工具失敗] {error_msg}")
        return error_msg

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
        
    print(f"\n🔍 [工具內部接收] 框架實際傳入: '{query_str}'\n")

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
# 四、 封裝 CrewAI RAG 啟動流程的通用函式
# =========================================================================
def ask_knowledge_crew(questions: List[str]) -> str:
    """
    接收一個動態的問題字串列表，自動初始化 CrewAI 並回傳最終解答。
    
    參數:
        questions (List[str]): 使用者提出的動態問題陣列。
        
    回傳:
        str: CrewAI 整合知識庫後輸出的最終解答文字。
    """
    if not questions:
        return "錯誤：傳入的問題列表為空。"

    # 1. 動態生成符合題數的 Prompt 字串
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    total_count = len(questions)

    # 2. 初始化本地 LLM
    internal_ollama_llm = LLM(
        # temperature=0,
        model="ollama/llama3.2:latest",
        base_url=OLLAMA_URL
    )

    # 3. 建立專屬 Agent（加入動態目標描述）
    hr_support_agent = Agent(
        role="高級智囊顧問",
        goal=f"你必須針對目前提出的 {total_count} 個問題，逐一使用工具查詢並提供完整解答。",
        backstory=f"""你是一位極度嚴謹且非常有耐心的助理。
        當使用者丟給你多個問題時，你絕對不能偷懶。
        你必須針對【這 {total_count} 個問題中的每一個】，單獨呼叫【query_knowledge_tool】工具。
        直到所有題目都有答案後，才能給出最終回覆。""",
        tools=[query_knowledge_tool],
        llm=internal_ollama_llm,
        max_iter=5, # 保持高迭代，防止中途放棄
        verbose=True
    )

    # 4. 建立動態 Task
    task1 = Task(
        description=f"""有一位新進員工詢問了以下問題，你必須分步驟處理：

【待回答的問題列表】：
{formatted_questions}

【核心執行步驟與鐵律】：
1. **逐題拆解分析**：請先確認這裡總共有 {total_count} 個問題。你絕對不能一次把所有問題的大雜燴丟進工具查詢。
2. **獨立關鍵字檢索**：針對上述列表中的【每一個獨立問題】，你必須單獨提取出該題的核心關鍵字，並針對每一題單獨呼叫至少一次【query_knowledge_tool】工具。
3. **分次收集上下文**：重複執行步驟 2，直到你為列表中的【每一道題目】都成功收集到對應的知識庫片段。
4. **嚴格核對**：在給出最終答案前，再次核對問題列表，確保回答的數量與問題數量完全一致。""",

        expected_output=f"""請務必嚴格依據檢索到的內容，以條列式格式回答所有問題。
必須確保最終輸出的答案編號與題目列表完全對應，共有 {total_count} 題就必須答滿 {total_count} 題，編號由 1 開始到 {total_count}，切勿自己發明內容。
若某題在工具中完全查無資料，請老實回答『未提及』。""",
        agent=hr_support_agent
    )

    # 5. 組裝 Crew
    crew = Crew(
        agents=[hr_support_agent],
        tasks=[task1],
        verbose=True
    )

    print(f"\n🚀 [CrewAI 啟動] 正在為您處理共 {total_count} 個動態問題...")
    
    # 6. 執行並返回結果字串
    crew_output = crew.kickoff()
    return str(crew_output)


# =========================================================================
# 五、 實際在程式碼中測試與調用（範例）
# =========================================================================
if __name__ == "__main__":
    
    # # 範例情境 A：今天想問 6 個問題
    # test_questions_1 = [
    #     "台鐵交通費用要如何報銷？",
    #     "報銷期限是幾個工作天提交系統？",
    #     "這台閘道器的處理器規格和時脈是多少？",
    #     "如果工廠設備只支援 Modbus RTU 且環境高達 70°C，這台設備能用嗎？要怎麼把資料傳上雲端？",
    #     "這款閘道器可以放在室外淋雨環境使用嗎？",
    #     "這台設備的額定輸入電壓（電壓範圍）是多少？"
    # ]
    #
    # print("\n--- 執行測試 A：多重混合問題 ---")
    # final_answer_A = ask_knowledge_crew(test_questions_1)
    # print("\n=== 最終輸出結果 A ===")
    # print(final_answer_A)
    
    
    # 範例情境 B：明天只想單獨問 2 個產品規格問題，完全不需要修改任何 Task/Agent 提示詞！
    test_questions_2 = [
        "這款閘道器可以放在室外淋雨環境使用嗎？",
        "這台閘道器的處理器時脈是多少？",
        "這台設備的額定輸入電壓（電壓範圍）是多少？"
    ]
    
    print("\n--- 執行測試 B：純產品規格問題 ---")
    final_answer_B = ask_knowledge_crew(test_questions_2)
    print("\n=== 最終輸出結果 B ===")
    print(final_answer_B)
