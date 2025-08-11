from __future__ import annotations

# =============================================================================
# common_utils.base_dialogue_agent (migrated to shared_utils)
# =============================================================================
"""Reusable base class for persona-based Socratic dialogue agents.

This module extracts the core workflow from role-based agents so that future
agents can be created by subclassing :class:`BaseDialogueAgent` with only
minimal configuration.
"""

from typing import Any, Dict, List, Optional, TypedDict
import os
import re

from langchain_community.vectorstores import FAISS
from langchain_dashscope.embeddings import DashScopeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from .llm_wrapper import CustomChatDashScope


class DialogueGraphState(TypedDict):
    """TypedDict describing the state as it flows through LangGraph."""

    user_input: str
    current_topic: str
    simulated_character: str
    conversation_history: List[Dict[str, str]]
    retrieved_docs: List[str]
    socratic_response: str
    turn_count: int
    error_message: Optional[str]
    dialogue_status: str  # "continue", "end", "error"


class BaseDialogueAgent:
    """Base class encapsulating the Socratic-dialogue workflow."""

    _INTENT_PROMPT_TMPL = PromptTemplate.from_template(
        """
用户希望进行一场关于{subject_name}的苏格拉底式对话，并希望我模仿特定人物的语气。
请从用户的输入中识别出“对话主题”和“希望模仿的历史人物”。
如果未明确指定人物，请默认“{default_character}”。如果未明确指定主题，请默认“{default_topic}”。

请以 JSON 格式输出，例如：
{{
    "topic": "实践与认识的关系",
    "character": "马克思"
}}

用户输入: {{user_input}}
        """
    )

    def __init__(
        self,
        *,
        subject_name: str,
        vectorstore_path: str,
        default_topic: str = "马克思主义哲学",
        default_character: str = "马克思",
        llm_model: str = "qwen-max",
        temperature: float = 0.8,
        embedding_model: str = "text-embedding-v2",
    ) -> None:
        self.subject_name = subject_name
        self.vectorstore_path = vectorstore_path
        self.default_topic = default_topic
        self.default_character = default_character

        if "DASHSCOPE_API_KEY" not in os.environ:
            raise EnvironmentError("Please set the DASHSCOPE_API_KEY environment variable.")

        try:
            self.embeddings = DashScopeEmbeddings(model=embedding_model)
            self.llm = CustomChatDashScope(model=llm_model, temperature=temperature)
            print(f"[{self.subject_name}] LLM & Embedding models initialised.")
        except Exception as exc:
            raise RuntimeError(f"Model initialisation failed: {exc}") from exc

        self.vectorstore = self._load_knowledge_base()
        self.graph = self._build_graph()

        self.generation_kwargs: dict = {}
        self.retrieval_k: int = 5

    def set_generation_params(self, *, max_tokens: int | None = None, timeout: int | None = None, retrieval_k: int | None = None) -> None:
        if max_tokens is not None:
            self.generation_kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            self.generation_kwargs["timeout"] = timeout
        if retrieval_k is not None and retrieval_k > 0:
            self.retrieval_k = retrieval_k

    def _load_knowledge_base(self) -> Optional[FAISS]:
        try:
            print(f"[{self.subject_name}] Loading knowledge base ...")
            return FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            print(f"[{self.subject_name}] ⚠️  Failed to load knowledge base: {exc}. Running without retrieval.")
            return None

    def parse_user_intent_node(self, state: DialogueGraphState) -> Dict[str, Any]:
        print("Parsing user intent ...")

        user_input = state["user_input"]
        current_topic = state["current_topic"]
        simulated_character = state["simulated_character"]

        if state["turn_count"] == 0:
            intent_prompt = self._INTENT_PROMPT_TMPL.format(
                user_input=user_input,
                default_topic=self.default_topic,
                default_character=self.default_character,
                subject_name=self.subject_name,
            )
            messages = [
                SystemMessage(content="你是一个意图识别专家。"),
                HumanMessage(content=intent_prompt),
            ]

            try:
                llm_response = self.llm.invoke(messages)
                match = re.search(r"\{.*\}", llm_response.content, re.DOTALL)
                if match:
                    parsed = eval(match.group(0))  # controlled LLM output
                    current_topic = parsed.get("topic", self.default_topic)
                    simulated_character = parsed.get("character", self.default_character)
                else:
                    raise ValueError("LLM did not return valid JSON.")
            except Exception as exc:
                print(f"Intent parsing failed: {exc} – falling back to defaults.")
                current_topic = self.default_topic
                simulated_character = self.default_character

            conversation_history = [{"role": "user", "content": user_input}]
        else:
            conversation_history = state["conversation_history"] + [{"role": "user", "content": user_input}]

        return {
            "current_topic": current_topic,
            "simulated_character": simulated_character,
            "conversation_history": conversation_history,
            "error_message": None,
            "dialogue_status": "continue",
        }

    def retrieve_knowledge_node(self, state: DialogueGraphState) -> Dict[str, Any]:
        print(f"Retrieving docs for topic '{state['current_topic']}' ...")

        if self.vectorstore is None:
            err = "Vector store not loaded."
            print(err)
            return {"retrieved_docs": [], "error_message": err, "dialogue_status": "error"}

        try:
            query = f"{state['current_topic']} {self.subject_name} {state['simulated_character']}"
            docs = self.vectorstore.similarity_search(query, k=self.retrieval_k)
            retrieved = list(dict.fromkeys([doc.page_content for doc in docs]))[:5]
            print(f"Retrieved {len(retrieved)} document snippets.")
            return {"retrieved_docs": retrieved, "error_message": None, "dialogue_status": "continue"}
        except Exception as exc:
            err = f"Retrieval error: {exc}"
            print(err)
            return {"retrieved_docs": [], "error_message": err, "dialogue_status": "error"}

    def generate_socratic_response_node(self, state: DialogueGraphState) -> Dict[str, Any]:
        print("Generating Socratic response ...")

        current_topic = state["current_topic"]
        simulated_character = state["simulated_character"]
        conversation_history = state["conversation_history"]
        retrieved_docs = state["retrieved_docs"]

        system_content = (
            f"你是一个资深的{self.subject_name}教师，现在你正在扮演 {simulated_character}，与学生进行一场关于 {current_topic} 的苏格拉底式对话。\n"
            "你的目标是：\n"
            "1. 模仿 {simulated_character} 的说话语气、风格和常用词汇。\n"
            "2. 保持苏格拉底式对话的核心：不直接给出答案，而是通过一系列启发性的问题引导学生思考。\n"
            "3. 问题应基于当前对话内容和参考资料，聚焦并促进思考。\n"
            "4. 如果学生回答偏离主题，尝试巧妙引导回主题。\n"
            "5. 当你认为学生对某个概念已经有了足够深入的思考时，可适当总结或提出更高层次的问题。\n"
            "6. 使用简洁的分点与小标题组织语言，突出层次与逻辑，必要处使用**加粗**强调关键词。\n\n"
            "参考资料：\n"
            f"{'   '.join(retrieved_docs)}\n\n"
            "当前对话历史：\n"
        )
        messages: List[AIMessage | HumanMessage | SystemMessage] = [SystemMessage(content=system_content)]
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        try:
            response = self.llm.invoke(messages, **self.generation_kwargs)
            ai_text = response.content
            new_history = conversation_history + [{"role": "assistant", "content": ai_text}]
            return {
                "socratic_response": ai_text,
                "conversation_history": new_history,
                "error_message": None,
                "turn_count": state["turn_count"] + 1,
                "dialogue_status": "continue",
            }
        except Exception as exc:
            err = f"Generation error: {exc}"
            print(err)
            return {
                "socratic_response": "抱歉，生成回应时出现问题。请稍后再试。",
                "error_message": err,
                "dialogue_status": "error",
            }

    def _build_graph(self):
        print(f"[{self.subject_name}] Building workflow graph ...")
        workflow = StateGraph(DialogueGraphState)
        workflow.add_node("parse_user_intent", self.parse_user_intent_node)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge_node)
        workflow.add_node("generate_socratic_response", self.generate_socratic_response_node)

        workflow.set_entry_point("parse_user_intent")
        workflow.add_edge("parse_user_intent", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "generate_socratic_response")
        workflow.add_edge("generate_socratic_response", END)

        print(f"[{self.subject_name}] Workflow graph compiled.")
        return workflow.compile()

    def process_dialogue(
        self,
        user_input: str,
        current_state: Optional[DialogueGraphState] = None,
    ) -> Dict[str, Any]:
        print(f"\n>> USER: {user_input}")
        if self.graph is None:
            return {"response": "Graph not available", "status": "error"}

        if current_state is None:
            init_state: DialogueGraphState = {
                "user_input": user_input,
                "current_topic": "",
                "simulated_character": "",
                "conversation_history": [],
                "retrieved_docs": [],
                "socratic_response": "",
                "turn_count": 0,
                "error_message": None,
                "dialogue_status": "continue",
            }
        else:
            init_state = {
                "user_input": user_input,
                "current_topic": current_state["current_topic"],
                "simulated_character": current_state["simulated_character"],
                "conversation_history": current_state["conversation_history"],
                "retrieved_docs": current_state["retrieved_docs"],
                "socratic_response": "",
                "turn_count": current_state["turn_count"],
                "error_message": None,
                "dialogue_status": "continue",
            }

        try:
            final_state = self.graph.invoke(init_state)
            if final_state["error_message"]:
                return {
                    "response": f"Error: {final_state['error_message']}",
                    "status": "error",
                    "state": final_state,
                }
            return {
                "response": final_state["socratic_response"],
                "status": "continue",
                "state": final_state,
            }
        except Exception as exc:
            return {
                "response": f"System error: {exc}",
                "status": "error",
                "state": init_state,
            }


