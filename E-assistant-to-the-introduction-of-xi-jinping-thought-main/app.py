import os
import sys
from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
import base64
import tempfile
import uuid
from dotenv import load_dotenv

"""
将 shared_utils 当作普通文件夹使用：
把工作区根目录与 `启真问智项目汇总` 加入 sys.path，
确保 `启真问智项目汇总/shared_utils` 可被直接导入。
"""
try:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    _INNER_SHARED_PARENT = os.path.join(_PROJECT_ROOT, "启真问智项目汇总")
    if os.path.isdir(_INNER_SHARED_PARENT) and _INNER_SHARED_PARENT not in sys.path:
        sys.path.insert(0, _INNER_SHARED_PARENT)
    # repo root (one level above project root)
    _REPO_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, os.pardir))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
except Exception:
    pass

from typing import Optional

try:
    from shared_utils.base_kg_agent import BaseKnowledgeGraphAgent
    from shared_utils.base_retrieval_agent import BaseRetrievalAgent
    from shared_utils.base_agent import BaseAgent
except Exception:
    # 兼容旧结构
    from common_utils.base_kg_agent import BaseKnowledgeGraphAgent
    from common_utils.base_retrieval_agent import BaseRetrievalAgent
    from common_utils.base_agent import BaseAgent

try:
    from shared_utils.multimodal_agent import MayuanMultimodalAgent
except Exception:
    from common_utils.multimodal_agent import MayuanMultimodalAgent
from role_agent import SocratesAgent


class XigaiKnowledgeGraphAgent(BaseKnowledgeGraphAgent):
    def __init__(self):
        super().__init__(subject_name="习近平新时代中国特色社会主义思想概论", vectorstore_path="database_agent_xigai")


class XigaiQuestionAgent(BaseAgent):
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
        except Exception:
            self.multimodal_agent = None
        self._last_full_output = ""
        self._last_question_only_output = ""

    def process_request(self, user_input: str) -> str:
        if any(kw in user_input for kw in ["解析", "答案", "讲解", "答案解析", "参考答案"]):
            return self._last_full_output or "当前没有可供解析的题目，请先提出出题需求。"
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
        filtered = []
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
        import re as _re
        result = "\n".join(filtered)
        result = _re.sub(r"\n{3,}", "\n\n", result).strip("\n")
        return result


class XigaiAnswerAgent(BaseRetrievalAgent):
    def __init__(self):
        super().__init__(
            subject_name="习近平新时代中国特色社会主义思想概论",
            vectorstore_path="database_agent_xigai",
            llm_model="qwen-max",
            embedding_model="text-embedding-v2",
            temperature=0.3,
        )
        try:
            self.multimodal_agent = MayuanMultimodalAgent()
        except Exception:
            self.multimodal_agent = None

    def process_multimodal_request(self, text_input: str, image_path: Optional[str] = None) -> str:
        if not image_path:
            return self.process_request(text_input)
        try:
            if self.multimodal_agent:
                return self.multimodal_agent.process_multimodal_request(text_input, image_path)
        except Exception:
            pass
        return self.process_request(text_input)

    def process_request(self, user_question: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        docs = self._retrieve_docs(f"{user_question} {self.subject_name}", k=5)
        context = "\n\n".join(docs[:5])
        prompt = (
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
            import re
            answer = str(getattr(response, "content", response)).strip()
            answer = re.sub(r"^`+|`+$", "", answer).strip()
            return answer
        except Exception:
            return "抱歉，回答过程中出现问题，请稍后再试。"


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', tempfile.gettempdir())
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def save_uploaded_image(image_data: str) -> Optional[str]:
    try:
        mime_type = None
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            header, data = image_data.split(',', 1)
            try:
                mime_type = header.split(';')[0].split(':', 1)[1]
            except Exception:
                mime_type = None
            image_binary = base64.b64decode(data)
        else:
            image_binary = base64.b64decode(image_data)
        if len(image_binary) > 16 * 1024 * 1024:
            return None
        try:
            img = Image.open(BytesIO(image_binary))
            img.verify()
            real_format = img.format
            if img.width > 4096 or img.height > 4096:
                return None
        except Exception:
            return None
        def guess_ext(mt: Optional[str], fmt: Optional[str]) -> str:
            mapping = { 'JPEG': 'jpg', 'JPG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'BMP': 'bmp', 'GIF': 'gif' }
            if fmt and fmt.upper() in mapping:
                return mapping[fmt.upper()]
            if mt and '/' in mt:
                mt_ext = mt.split('/')[-1].lower()
                return 'jpg' if mt_ext == 'jpeg' else mt_ext
            return 'jpg'
        ext = guess_ext(mime_type, locals().get('real_format'))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
        temp_file.write(image_binary)
        temp_file.close()
        return temp_file.name
    except Exception:
        return None


# -------------- Agent 初始化 --------------
try:
    question_agent = XigaiQuestionAgent()
except Exception as e:
    print(f"Error loading XigaiQuestionAgent: {e}")
    question_agent = None

try:
    kg_agent = XigaiKnowledgeGraphAgent()
except Exception as e:
    print(f"Error loading XigaiKnowledgeGraphAgent: {e}")
    kg_agent = None

try:
    qa_agent = XigaiAnswerAgent()
except Exception as e:
    print(f"Error loading XigaiAnswerAgent: {e}")
    qa_agent = None

# ----- Role Play Agent -----
dialogue_sessions = {}
try:
    socrates_agent = SocratesAgent()
    print("SocratesAgent initialized (Xigai context).")
except Exception as e:
    print(f"Error initializing SocratesAgent: {e}")
    socrates_agent = None


@app.route('/chat_ui')
def chat_ui():
    return render_template('index.html')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/role')
def role_chat_page():
    return render_template('role_chat.html')


@app.route('/start_dialogue', methods=['POST'])
def start_dialogue():
    if not socrates_agent:
        return jsonify({"error": "AI助手未正确初始化"}), 500
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    image_data = data.get("image")
    response_mode = (data.get("response_mode") or "balanced").lower()

    if not user_message and not image_data:
        return jsonify({"error": "请输入您想探讨的话题或上传图片"}), 400

    image_path = None
    if image_data:
        image_path = save_uploaded_image(image_data)
        if not image_path:
            return jsonify({"error": "图片处理失败"}), 400

    try:
        session_id = str(uuid.uuid4())
        try:
            if hasattr(socrates_agent, 'set_generation_params'):
                if response_mode == 'fast':
                    socrates_agent.set_generation_params(max_tokens=400, timeout=15, retrieval_k=3)
                elif response_mode == 'detailed':
                    socrates_agent.set_generation_params(max_tokens=1600, timeout=45, retrieval_k=7)
                else:
                    socrates_agent.set_generation_params(max_tokens=1000, timeout=30, retrieval_k=5)
        except Exception:
            pass

        if not user_message:
            user_message = "请结合这张图片开始对话并提出苏格拉底式问题。"
        if image_path and hasattr(socrates_agent, 'process_multimodal_dialogue'):
            response_data = socrates_agent.process_multimodal_dialogue(user_message, None, image_path)
        else:
            response_data = socrates_agent.process_dialogue(user_message, None)

        if response_data.get("status") == "error":
            return jsonify({"error": response_data.get("response", "内部错误")}), 500
        dialogue_sessions[session_id] = response_data["state"]
        return jsonify({
            "session_id": session_id,
            "response": response_data["response"],
            "character": response_data["state"]["simulated_character"],
            "topic": response_data["state"]["current_topic"],
            "turn_count": response_data["state"]["turn_count"],
        })
    finally:
        if image_path:
            try:
                if os.path.exists(image_path):
                    os.unlink(image_path)
            except Exception:
                pass


@app.route('/continue_dialogue', methods=['POST'])
def continue_dialogue():
    if not socrates_agent:
        return jsonify({"error": "AI助手未正确初始化"}), 500
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    user_message = (data.get("message") or "").strip()
    image_data = data.get("image")
    response_mode = (data.get("response_mode") or "balanced").lower()

    if not session_id or session_id not in dialogue_sessions:
        return jsonify({"error": "会话已过期，请重新开始对话"}), 400
    if not user_message and not image_data:
        return jsonify({"error": "请输入您的回应或上传图片"}), 400

    image_path = None
    if image_data:
        image_path = save_uploaded_image(image_data)
        if not image_path:
            return jsonify({"error": "图片处理失败"}), 400

    try:
        current_state = dialogue_sessions[session_id]
        try:
            if hasattr(socrates_agent, 'set_generation_params'):
                if response_mode == 'fast':
                    socrates_agent.set_generation_params(max_tokens=400, timeout=15, retrieval_k=3)
                elif response_mode == 'detailed':
                    socrates_agent.set_generation_params(max_tokens=1600, timeout=45, retrieval_k=7)
                else:
                    socrates_agent.set_generation_params(max_tokens=1000, timeout=30, retrieval_k=5)
        except Exception:
            pass

        if not user_message:
            user_message = "请结合这张图片继续对话并提出苏格拉底式问题。"
        if image_path and hasattr(socrates_agent, 'process_multimodal_dialogue'):
            response_data = socrates_agent.process_multimodal_dialogue(user_message, current_state, image_path)
        else:
            response_data = socrates_agent.process_dialogue(user_message, current_state)

        if response_data.get("status") == "error":
            return jsonify({"error": response_data.get("response", "内部错误")}), 500
        dialogue_sessions[session_id] = response_data["state"]
        return jsonify({
            "response": response_data["response"],
            "character": response_data["state"]["simulated_character"],
            "topic": response_data["state"]["current_topic"],
            "turn_count": response_data["state"]["turn_count"],
        })
    finally:
        if image_path:
            try:
                if os.path.exists(image_path):
                    os.unlink(image_path)
            except Exception:
                pass


@app.route('/end_dialogue', methods=['POST'])
def end_dialogue():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if session_id and session_id in dialogue_sessions:
        dialogue_sessions.pop(session_id, None)
        return jsonify({"message": "对话已结束"})
    return jsonify({"message": "会话未找到或已结束"})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    image_data = data.get("image")
    response_mode = (data.get("response_mode") or "balanced").lower()

    if not user_message and not image_data:
        return jsonify({"error": "请输入文本或上传图片"}), 400

    image_path = None
    if image_data:
        image_path = save_uploaded_image(image_data)
        if not image_path:
            return jsonify({"error": "图片处理失败"}), 400

    response_text = ""
    try:
        if not user_message:
            user_message = "请结合图片进行分析并回答问题。"

        def apply_mode(agent):
            try:
                if not hasattr(agent, 'set_generation_params'):
                    return
                if response_mode == 'fast':
                    agent.set_generation_params(max_tokens=400, timeout=15, retrieval_k=3)
                elif response_mode == 'detailed':
                    agent.set_generation_params(max_tokens=1600, timeout=45, retrieval_k=7)
                else:
                    agent.set_generation_params(max_tokens=1000, timeout=30, retrieval_k=5)
            except Exception:
                pass

        if any(k in user_message for k in ["知识图谱", "思维导图", "mindmap", "图谱"]):
            if kg_agent:
                apply_mode(kg_agent)
                if image_path:
                    response_text = "知识图谱生成功能暂时不支持图片输入，请使用纯文本描述您需要的知识图谱主题。"
                else:
                    response_text = kg_agent.build_knowledge_graph(user_message)
            else:
                response_text = "知识图谱助手未成功加载，无法处理您的请求。"
        else:
            answer_keywords = ["解答", "答案", "解析", "请回答", "帮我回答", "帮我解答"]
            contains_mcq_options = False
            try:
                import re
                if re.search(r"[A-DＡ-Ｄ][\.．、]\s?", user_message):
                    contains_mcq_options = True
            except Exception:
                contains_mcq_options = False

            if any(kw in user_message for kw in answer_keywords) or contains_mcq_options:
                if qa_agent:
                    apply_mode(qa_agent)
                    if image_path and hasattr(qa_agent, 'process_multimodal_request'):
                        response_text = qa_agent.process_multimodal_request(user_message, image_path)
                    else:
                        response_text = qa_agent.process_request(user_message)
                else:
                    response_text = "问答助手未成功加载，无法处理您的请求。"
            else:
                if question_agent:
                    apply_mode(question_agent)
                    if hasattr(question_agent, 'process_multimodal_request') and image_path:
                        response_text = question_agent.process_multimodal_request(user_message, image_path)
                    else:
                        if image_path:
                            response_text = "当前版本暂时不支持图片分析，请使用纯文本提问。"
                        else:
                            response_text = question_agent.process_request(user_message)
                else:
                    response_text = "出题助手未成功加载，无法处理您的请求。"
    except Exception as e:
        response_text = f"处理您的请求时发生内部错误: {e}"
    finally:
        try:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
        except Exception:
            pass

    return jsonify({"response": response_text})


def run_app():
    load_dotenv()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("\n" + "="*50)
        print("❌ 警告: 未找到 DASHSCOPE_API_KEY 环境变量。")
        print("如果您的 Agent 需要调用 DashScope API，请务必进行设置。")
        print("您可以临时在终端运行:")
        print("  $env:DASHSCOPE_API_KEY='your_api_key_here'  (Windows PowerShell)")
        print("  export DASHSCOPE_API_KEY='your_api_key_here' (Linux/Mac)")
        print("="*50 + "\n")
    app.run(debug=True, port=5041)


if __name__ == '__main__':
    run_app()



