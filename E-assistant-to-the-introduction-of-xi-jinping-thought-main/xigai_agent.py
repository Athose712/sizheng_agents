"""
习近平新时代中国特色社会主义思想概论 智能出题 Agent
"""
import os
import sys
from typing import Optional

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
    from shared_utils.base_agent import BaseAgent
    from shared_utils.multimodal_agent import MayuanMultimodalAgent
except Exception:
    from common_utils.base_agent import BaseAgent
    from common_utils.multimodal_agent import MayuanMultimodalAgent


class XigaiQuestionAgent(BaseAgent):
    """“习近平新时代中国特色社会主义思想概论”课程的智能出题 Agent。"""

    def __init__(self):
        common_topics = [
            "新时代中国特色社会主义思想", "两个确立", "两个维护", "五位一体总体布局", "四个全面战略布局",
            "新发展理念", "全面深化改革", "全面依法治国", "全面从严治党", "共同富裕",
        ]

        super().__init__(
            subject_name="习近平新时代中国特色社会主义思想概论",
            default_topic="新时代中国特色社会主义思想",
            common_topics=common_topics,
            vectorstore_path="database_agent_xigai",
        )

        try:
            self.multimodal_agent = MayuanMultimodalAgent()
        except Exception as e:
            print(f"[习概Agent] 多模态功能初始化失败: {e}")
            self.multimodal_agent = None

        self._last_full_output: str = ""
        self._last_question_only_output: str = ""

    def process_request(self, user_input: str) -> str:
        if any(kw in user_input for kw in ["解析", "答案", "讲解", "答案解析", "参考答案"]):
            if self._last_full_output:
                return self._last_full_output
            return "当前没有可供解析的题目，请先提出出题需求。"
        full_output = super().process_request(user_input)
        self._last_full_output = full_output
        self._last_question_only_output = self._strip_explanations(full_output)
        return self._last_question_only_output

    def process_multimodal_request(self, text_input: str, image_path: Optional[str] = None) -> str:
        if not image_path or not self.multimodal_agent:
            return self.process_request(text_input)
        if any(kw in text_input for kw in ["解析", "答案", "讲解", "答案解析", "参考答案"]):
            return self._last_full_output or "当前没有可供解析的题目，请先提出出题需求。"
        try:
            full_output = self.multimodal_agent.process_multimodal_request(text_input, image_path)
            self._last_full_output = full_output
            if any(kw in text_input for kw in ["出题", "生成题目", "题目", "选择题", "判断题", "简答题", "试题", "练习"]):
                self._last_question_only_output = self._strip_explanations(full_output)
                return self._last_question_only_output
            return full_output
        except Exception:
            return self.process_request(text_input)

    def _strip_explanations(self, text: str) -> str:
        import re
        lines = text.splitlines()
        filtered: list[str] = []

        start_patterns = [
            r"^\s*(?:正确?答案|参考答案|标准答案|答案解析|解析|解答|讲解|答案是|答案为|Answer|Explanation)\s*[:：】\])]?.*$",
            r"^\s*[（(【\[]?(?:答|解)\s*[：:]\s*.*$",
            r"^\s*[【\[]?(?:答案|解析|参考答案)[】\]]\s*.*$",
        ]
        inline_patterns = [r"(正确?答案|参考答案|标准答案|答案解析|解析|解答|答案是|答案为|Answer|Explanation)\s*[:：]?\s*"]
        boundary_patterns = [
            r"^\s*(?:题目|选择题|判断题|简答题)\s*\d+",
            r"^\s*(?:选择题|判断题|简答题)\s*[：:]\s*$",
            r"^\s*\d+\s*[、\.\)．]",
        ]
        start_regexes = [re.compile(p, re.IGNORECASE) for p in start_patterns]
        inline_regexes = [re.compile(p, re.IGNORECASE) for p in inline_patterns]
        boundary_regexes = [re.compile(p) for p in boundary_patterns]
        in_strip_block = False

        def is_start(line: str) -> bool:
            return any(r.match(line) for r in start_regexes) or any(r.search(line) for r in inline_regexes)

        def is_boundary(line: str) -> bool:
            return any(r.match(line) for r in boundary_regexes)

        for line in lines:
            if in_strip_block:
                if is_boundary(line):
                    in_strip_block = False
                    filtered.append(line)
                else:
                    continue
            else:
                if is_start(line):
                    in_strip_block = True
                    continue
                filtered.append(line)

        result = "\n".join(filtered)
        result = re.sub(r"\n{3,}", "\n\n", result).strip("\n")
        return result



