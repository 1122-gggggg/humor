import os
import sys
from pathlib import Path
from tqdm import tqdm

def extract_text_from_pdf(pdf_path: str) -> str:
    """從 PDF 檔中抽取文字。支援 PyMuPDF (fitz) 或 PyPDF2。"""
    text = ""
    try:
        import fitz  # PyMuPDF
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except ImportError:
            print("❌ 缺少 PDF 解析套件！請執行: pip install PyMuPDF 或 pip install PyPDF2")
            print("建議使用 PyMuPDF，速度較快且精準。")
            sys.exit(1)
        except Exception as e:
             print(f"⚠️ 解析 {pdf_path} 時發生錯誤 (PyPDF2): {e}")
    except Exception as e:
        print(f"⚠️ 解析 {pdf_path} 時發生錯誤 (PyMuPDF): {e}")
        
    return text

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """將長文本切割成重疊的 chunks (Token/字元級簡單切割)"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def build_vectordb(pdf_dir: str, db_dir: str):
    print(f"📚 正在掃描目錄: {pdf_dir}")
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    if not pdf_files:
        print("❌ 找不到任何 PDF 檔案！請確認路徑。")
        return
        
    print(f"🔍 找到 {len(pdf_files)} 份教材，準備萃取知識並寫入 ChromaDB...")
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("❌ 缺少向量資料庫套件！請執行: pip install chromadb")
        sys.exit(1)
        
    # 初始化 ChromaDB
    client = chromadb.PersistentClient(path=db_dir)
    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3" # 這裡與教練系統使用的 SOTA 保持一致
    )
    
    collection = client.get_or_create_collection(
        name="comedy_rules",
        embedding_function=default_ef,
        metadata={"description": "單口喜劇教材與實戰心法"}
    )
    
    # 逐一處理並嵌入
    for pdf_path in tqdm(pdf_files, desc="處理教材中"):
        title = pdf_path.stem
        # 1. 抽取文字
        text = extract_text_from_pdf(str(pdf_path))
        if not text.strip():
            print(f"\n⚠️ 警告: {title} 抽取不到文字 (可能是純圖檔掃描)，先跳過。")
            continue
            
        # 2. 切片 (RAG 的關鍵)
        chunks = create_chunks(text, chunk_size=600, overlap=100)
        
        # 3. 準備寫入格式
        ids = [f"{title}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": title, "chunk_index": i} for i in range(len(chunks))]
        documents = chunks
        
        # 4. 寫入資料庫
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    print(f"\n✅ 知識庫建置完成！共寫入了 {collection.count()} 個知識區塊。")
    print(f"資料庫已儲存至: {db_dir}")

if __name__ == "__main__":
    SOURCE_DIR = r"C:\Users\90607\OneDrive\桌面\段子\表演"
    DB_OUT_DIR = r"data\chroma_db"
    
    os.makedirs(DB_OUT_DIR, exist_ok=True)
    build_vectordb(SOURCE_DIR, DB_OUT_DIR)
