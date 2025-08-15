import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_dashscope.embeddings import DashScopeEmbeddings

# 注意：请通过环境变量提供 DASHSCOPE_API_KEY

RAW_DIR = "./maogai_raw_data/"
DB_DIR = "database_agent_maogai"

print(f"正在从 '{RAW_DIR}' 文件夹加载文档...")
loader = PyPDFDirectoryLoader(RAW_DIR)
documents = loader.load()
print(f"成功加载 {len(documents)} 份文档。")

# 过滤空页面；若为空则尝试用 PyMuPDF 再解析
documents = [d for d in documents if getattr(d, 'page_content', '') and d.page_content.strip()]
if not documents:
    print("检测到解析文本为空，尝试使用 PyMuPDF 进行二次解析...")
    import glob
    import os as _os
    pdf_paths = glob.glob(_os.path.join(RAW_DIR, '*.pdf'))
    mu_docs = []
    for p in pdf_paths:
        try:
            mu_docs.extend(PyMuPDFLoader(p).load())
        except Exception as e:
            print(f"PyMuPDF 解析失败: {p} -> {e}")
    documents = [d for d in mu_docs if getattr(d, 'page_content', '') and d.page_content.strip()]
    print(f"PyMuPDF 解析后有效文档数：{len(documents)}")
    if not documents:
        raise SystemExit("❌ 无可用文本内容：PDF 可能是纯图片，请先做 OCR 后再生成向量库。")

print("正在分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=1000,
	chunk_overlap=100
)
split_docs = text_splitter.split_documents(documents)
# 过滤空块
split_docs = [d for d in split_docs if getattr(d, 'page_content', '') and d.page_content.strip()]
print(f"文档被分割成 {len(split_docs)} 个小块。")

print("正在初始化文本嵌入模型...")
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

print("正在创建并保存向量数据库 (采用分批处理模式)...")
if not split_docs:
    raise SystemExit("❌ 分割后无文本块，无法构建向量库。")
batch_size = 50
vectorstore = FAISS.from_documents(split_docs[:batch_size], embeddings)
for i in range(batch_size, len(split_docs), batch_size):
	end_index = min(i + batch_size, len(split_docs))
	batch_docs = split_docs[i:end_index]
	if not batch_docs:
		continue
	vectorstore.add_documents(batch_docs)

vectorstore.save_local(DB_DIR)
print(f"知识库构建完成，并已保存到本地 '{DB_DIR}' 文件夹。")


