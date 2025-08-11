import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_dashscope.embeddings import DashScopeEmbeddings

# 请在运行脚本前设置环境变量 DASHSCOPE_API_KEY
# Windows PowerShell:  $env:DASHSCOPE_API_KEY='your_api_key_here'
# Linux/Mac:          export DASHSCOPE_API_KEY='your_api_key_here'

RAW_DIR = os.path.join(os.path.dirname(__file__), "jindaishi_raw_data")
DB_DIR = os.path.join(os.path.dirname(__file__), "database_agent_jindaishi")

if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise EnvironmentError("未设置 DASHSCOPE_API_KEY 环境变量")

    if not os.path.isdir(RAW_DIR):
        os.makedirs(RAW_DIR, exist_ok=True)
        print(f"已创建原始资料目录: {RAW_DIR} (请放入近现代史 PDF 资料)")

    print(f"正在从 '{RAW_DIR}' 文件夹加载文档...")
    loader = PyPDFDirectoryLoader(RAW_DIR)
    documents = loader.load()
    print(f"成功加载 {len(documents)} 份文档。")

    if len(documents) == 0:
        print("未检测到 PDF 文档，请将资料放入该目录后重试。")
        raise SystemExit(0)

    print("正在分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    print(f"文档被分割成 {len(split_docs)} 个小块。")

    print("正在初始化文本嵌入模型...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    print("正在创建并保存向量数据库 (分批处理)...")
    batch_size = 20
    vectorstore = FAISS.from_documents(split_docs[:batch_size], embeddings)
    for i in range(batch_size, len(split_docs), batch_size):
        end_index = min(i + batch_size, len(split_docs))
        batch_docs = split_docs[i:end_index]
        print(f"正在处理第 {i + 1} 到 {end_index} 个文档块...")
        vectorstore.add_documents(batch_docs)

    # 确保保存目录存在；若路径含非 ASCII 导致保存失败，则使用临时 ASCII 目录回退
    os.makedirs(DB_DIR, exist_ok=True)
    try:
        vectorstore.save_local(DB_DIR)
    except Exception as e:
        print(f"直接保存到 '{DB_DIR}' 失败，尝试使用临时 ASCII 目录回退。错误: {e}")
        fallback_dir = os.path.join(tempfile.gettempdir(), "faiss_tmp_jindaishi")
        os.makedirs(fallback_dir, exist_ok=True)
        vectorstore.save_local(fallback_dir)
        for name in ("index.faiss", "index.pkl"):
            src = os.path.join(fallback_dir, name)
            dst = os.path.join(DB_DIR, name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print(f"已通过回退方式保存到: '{DB_DIR}'。如需彻底避免此问题，建议将项目迁移到仅包含 ASCII 字符的路径下。")
    print(f"知识库构建完成，并已保存到本地 '{DB_DIR}' 文件夹。")


