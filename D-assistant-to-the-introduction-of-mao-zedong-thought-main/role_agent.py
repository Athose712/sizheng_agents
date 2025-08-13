import os
import sys
from typing import Optional

"""确保将共享代码目录加入 sys.path（作为普通文件夹使用）"""
try:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    _INNER_SHARED_PARENT = os.path.join(_PROJECT_ROOT, "启真问智项目汇总")
    if os.path.isdir(_INNER_SHARED_PARENT) and _INNER_SHARED_PARENT not in sys.path:
        sys.path.insert(0, _INNER_SHARED_PARENT)
    _REPO_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, os.pardir))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
except Exception:
    pass

try:
    from shared_utils.base_dialogue_agent import BaseDialogueAgent, DialogueGraphState
    from shared_utils.multimodal_agent import SocratesMultimodalAgent
except Exception:
    from common_utils.base_dialogue_agent import BaseDialogueAgent, DialogueGraphState
    from common_utils.multimodal_agent import SocratesMultimodalAgent


class SocratesAgent(BaseDialogueAgent):
    def __init__(self):
        super().__init__(
            subject_name="毛泽东思想概论",
            vectorstore_path="database_agent_maogai",
            default_topic="毛泽东思想",
            default_character="毛泽东",
            llm_model="qwen-max",
            temperature=0.8,
        )
        try:
            self.multimodal_agent = SocratesMultimodalAgent(character="毛泽东", topic="毛泽东思想")
        except Exception:
            self.multimodal_agent = None

    def process_multimodal_dialogue(self, user_input: str, current_state: Optional[dict] = None, image_path: Optional[str] = None) -> dict:
        if not image_path or not self.multimodal_agent:
            return self.process_dialogue(user_input, current_state)
        try:
            if current_state:
                character = current_state.get("simulated_character", "毛泽东")
                topic = current_state.get("current_topic", "毛泽东思想")
                self.multimodal_agent.update_dialogue_context(character, topic)
            response = self.multimodal_agent.process_multimodal_request(user_input, image_path)
            if not current_state:
                new_state = {
                    "simulated_character": "毛泽东",
                    "current_topic": "毛泽东思想",
                    "turn_count": 1,
                    "conversation_history": [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response},
                    ],
                }
                return {"status": "success", "response": response, "state": new_state}
            current_state = dict(current_state)
            current_state["turn_count"] = current_state.get("turn_count", 0) + 1
            current_state.setdefault("conversation_history", []).extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response},
            ])
            return {"status": "success", "response": response, "state": current_state}
        except Exception:
            return self.process_dialogue(user_input, current_state)


