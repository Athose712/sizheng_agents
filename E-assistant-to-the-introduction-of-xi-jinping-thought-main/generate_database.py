import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_dashscope.embeddings import DashScopeEmbeddings

# 注意：请通过环境变量提供 DASHSCOPE_API_KEY

RAW_DIR = "./xigai_raw_data/"
DB_DIR = "database_agent_xigai"

print(f"正在从 '{RAW_DIR}' 文件夹加载文档...")
loader = PyPDFDirectoryLoader(RAW_DIR)
documents = loader.load()
print(f"成功加载 {len(documents)} 份文档。")

print("正在分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=1000,
	chunk_overlap=100
)
split_docs = text_splitter.split_documents(documents)
print(f"文档被分割成 {len(split_docs)} 个小块。")

print("正在初始化文本嵌入模型...")
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

print("正在创建并保存向量数据库 (采用分批处理模式)...")
batch_size = 20
vectorstore = FAISS.from_documents(split_docs[:batch_size], embeddings)
for i in range(batch_size, len(split_docs), batch_size):
	end_index = min(i + batch_size, len(split_docs))
	batch_docs = split_docs[i:end_index]
	vectorstore.add_documents(batch_docs)

vectorstore.save_local(DB_DIR)
print(f"知识库构建完成，并已保存到本地 '{DB_DIR}' 文件夹。")



