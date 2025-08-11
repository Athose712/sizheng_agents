from __future__ import annotations

"""
思想道德与法治 检索增强问答 Agent
"""

import os
import re
import sys
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

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
    from shared_utils.base_retrieval_agent import BaseRetrievalAgent
    from shared_utils.multimodal_agent import MayuanMultimodalAgent
except Exception:
    from common_utils.base_retrieval_agent import BaseRetrievalAgent
    from common_utils.multimodal_agent import MayuanMultimodalAgent


class SixiangDaodeFazhiAnswerAgent(BaseRetrievalAgent):
    def __init__(
        self,
        *,
        subject_name: str = "思想道德与法治",
        vectorstore_path: str = "database_agent_sixiangdaodefazhi",
        llm_model: str = "qwen-max",
        temperature: float = 0.3,
        embedding_model: str = "text-embedding-v2",
    ) -> None:
        super().__init__(
            subject_name=subject_name,
            vectorstore_path=vectorstore_path,
            llm_model=llm_model,
            embedding_model=embedding_model,
            temperature=temperature,
        )

        try:
            self.multimodal_agent = MayuanMultimodalAgent()
            print(f"[{self.subject_name}] AnswerAgent – multimodal support initialised.")
        except Exception as exc:
            print(f"[{self.subject_name}] ⚠️  Multimodal initialisation failed: {exc}")
            self.multimodal_agent = None

    def process_multimodal_request(self, text_input: str, image_path: Optional[str] = None) -> str:
        if not image_path:
            return self.process_request(text_input)
        extracted_text = self._extract_text_from_image(image_path)
        if self.multimodal_agent is None:
            if extracted_text:
                base = text_input or "请根据图片中的题目进行解答。"
                combined = base + "\n\n以下是 OCR 自动识别的图片文字，请据此解答：\n" + extracted_text[:1500]
                return self.process_request(combined)
            return self.process_request(text_input)
        combined_input = text_input
        if extracted_text:
            combined_input += "\n\n以下是 OCR 自动识别的图片文字，请据此解答：\n" + extracted_text[:1500]
        try:
            response = self.multimodal_agent.process_multimodal_request(combined_input, image_path)
            return str(response).strip()
        except Exception:
            if extracted_text:
                return self.process_request(extracted_text)
            return self.process_request(text_input)

    def _build_prompt(self, user_question: str, context: str) -> str:
        return (
            f"你是一位精通{self.subject_name}的教师，请以结构化 Markdown 输出，条理清晰、重点明确。"
            "请优先完整与准确，适度展开，避免冗余。若下方\"参考资料\"足够，请严格基于资料作答；否则结合你的专业知识回答。\n\n"
            "请按照以下结构组织你的回答（如不适用可省略某些小节）：\n"
            "### 核心结论\n"
            "- 用1-2句加粗给出直接答案或观点。\n\n"
            "### 关键要点\n"
            "- 3-6条要点，每条不超过两句，必要处使用**加粗关键词**。\n\n"
            "### 依据与推理\n"
            "- 用2-5句解释你的论证链条，可引用资料中的关键句（简洁转述）。\n\n"
            "### 示例或应用（可选）\n"
            "- 给出1个贴近教学的例子或场景来帮助理解。\n\n"
            "### 小结与延伸（可选）\n"
            "- 用1-2句总结，并提出1个进一步思考方向。\n\n"
            "参考资料（可能为空）：\n" + context + "\n\n学生问题：" + user_question + "\n回答："
        )

    def process_request(self, user_question: str) -> str:
        docs = self._retrieve_docs(f"{user_question} {self.subject_name}", k=5)
        context = "\n\n".join(docs[:5])
        prompt = self._build_prompt(user_question, context)
        messages = [
            SystemMessage(content=(
                f"你是一位严谨的{self.subject_name}解答专家。"
                "请使用结构化 Markdown（标题、列表、加粗）输出，层次清晰，美观易读；"
                "在保证准确性的前提下适度展开，覆盖关键要点。"
            )),
            HumanMessage(content=prompt),
        ]
        try:
            response = self.llm.invoke(messages, **self.generation_kwargs)
            answer = str(getattr(response, "content", response)).strip()
            answer = re.sub(r"^`+|`+$", "", answer).strip()
            return answer
        except Exception as exc:
            print(f"[{self.subject_name}] Answer generation failed: {exc}")
            return "抱歉，回答过程中出现问题，请稍后再试。"

    def _extract_text_from_image(self, image_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
            custom_cmd = os.environ.get("TESSERACT_CMD")
            if custom_cmd and os.path.exists(custom_cmd):
                pytesseract.pytesseract.tesseract_cmd = custom_cmd  # type: ignore[attr-defined]
            else:
                common_paths = [
                    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                    r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
                ]
                for p in common_paths:
                    if os.path.exists(p):
                        pytesseract.pytesseract.tesseract_cmd = p  # type: ignore[attr-defined]
                        break
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            return text.strip()
        except Exception:
            return ""


