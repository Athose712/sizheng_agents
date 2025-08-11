"""
中国近现代史纲要 知识图谱 Agent
"""
import os
import sys

try:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    _INNER_SHARED_PARENT = os.path.join(_PROJECT_ROOT, "启真问智项目汇总")
    if os.path.isdir(_INNER_SHARED_PARENT) and _INNER_SHARED_PARENT not in sys.path:
        sys.path.insert(0, _INNER_SHARED_PARENT)
except Exception:
    pass

try:
    from shared_utils.base_kg_agent import BaseKnowledgeGraphAgent
except Exception:
    from common_utils.base_kg_agent import BaseKnowledgeGraphAgent


class JindaishiKnowledgeGraphAgent(BaseKnowledgeGraphAgent):
    def __init__(self):
        super().__init__(
            subject_name="中国近现代史纲要",
            vectorstore_path="database_agent_jindaishi",
        )


if __name__ == "__main__":
    agent = JindaishiKnowledgeGraphAgent()
    while True:
        q = input("请输入知识点(或 exit 退出): ").strip()
        if q.lower() in {"exit", "quit", "q"}: break
        if not q: continue
        print(agent.build_knowledge_graph(q))


