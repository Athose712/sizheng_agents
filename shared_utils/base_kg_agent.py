"""
Reusable BaseKnowledgeGraphAgent for generating Mermaid mindmaps.
为各课程的“知识图谱”功能提供统一基类，支持无向量库降级。
"""
import os
import re
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_dashscope.embeddings import DashScopeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from .llm_wrapper import CustomChatDashScope


class BaseKnowledgeGraphAgent:
    """基础知识图谱 Agent：输出 Mermaid mindmap + 简要总结。

    特性：
    - 优先从向量库检索上下文；若向量库缺失或加载失败，则自动降级为零检索模式（仍可生成图）。
    - 统一的 Mermaid mindmap 输出模板与后处理，便于前端直接渲染。
    """

    def __init__(self, subject_name: str, vectorstore_path: str):
        self.subject_name = subject_name
        self.vectorstore_path = vectorstore_path

        # LLM 与 Embedding
        if "DASHSCOPE_API_KEY" not in os.environ:
            raise EnvironmentError("Please set the DASHSCOPE_API_KEY environment variable.")

        self.embeddings = DashScopeEmbeddings(model="text-embedding-v2")
        self.llm = CustomChatDashScope(model="qwen-max", temperature=0.5)

        # 尝试加载向量库；失败则降级为 None
        self.vectorstore = None
        try:
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"[KG] 警告：向量库未找到或加载失败（{self.vectorstore_path}）。将使用零检索模式。原因: {e}")

        self.graph_prompt = PromptTemplate.from_template(
            """
你是一位{subject_name}知识图谱专家。请利用提供的"参考资料"，围绕知识点"{topic}"构建一个 Mermaid mindmap（思维导图）格式的知识图谱，突出关键概念及其主要关系，并保持简洁易读。

输出要求：
1. mindmap 总节点不超过 15 个，层级不超过 3 级，保证图谱信息清晰、结构美观，便于学生学习和整理思路。
2. 先输出 Mermaid 源代码，必须使用如下代码块格式：
```mermaid
mindmap
  root(({topic}))
    概念1
      子概念A
    概念2
```
3. Mermaid 代码块结束后，换行再输出一段不超过 100 字的中文总结，对图谱内容进行简洁概括。
4. 除以上内容外，不要输出其他文字。

参考资料：
{context}
"""
        )

    def _retrieve_docs(self, topic: str, k: int = 5) -> List[str]:
        query = f"{topic} {self.subject_name}"
        if self.vectorstore is None:
            return []
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"[KG] 检索失败，将返回空上下文。原因: {e}")
            return []

    def _generate_mermaid(self, topic: str, context: str) -> str:
        prompt_text = self.graph_prompt.format(
            subject_name=self.subject_name, topic=topic, context=context
        )
        messages = [
            SystemMessage(content="你是一位精通知识图谱构建的学者。"),
            HumanMessage(content=prompt_text),
        ]
        response = self.llm.invoke(messages)
        return str(getattr(response, "content", response)).strip()

    def _format_mermaid_response(self, raw_output: str) -> str:
        raw_output = raw_output.strip()
        mermaid_match = re.search(r"```mermaid(.*?)```", raw_output, re.DOTALL)
        if mermaid_match:
            mermaid_code = mermaid_match.group(1).strip()
            summary = raw_output[mermaid_match.end():].strip()
        else:
            mermaid_code = raw_output.replace("```mermaid", "").replace("```", "")
            summary = ""
        formatted_output = f"```mermaid\n{mermaid_code}\n```"
        if summary:
            formatted_output += f"\n\n{summary}"
        return formatted_output.strip()

    def build_knowledge_graph(self, topic: str) -> str:
        docs = self._retrieve_docs(topic, k=5)
        context = "\n\n".join(docs)
        raw_output = self._generate_mermaid(topic, context)
        return self._format_mermaid_response(raw_output)