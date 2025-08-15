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

from role_agent import SocratesAgent
from xigai_agent import XigaiQuestionAgent
from xigai_kg_agent import XigaiKnowledgeGraphAgent
from xigai_qa_agent import XigaiAnswerAgent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from shared_utils.llm_wrapper import CustomChatDashScope as _KGLLM


class KGAgentWrapper:
    def __init__(self):
        self._agent = None
        try:
            self._agent = XigaiKnowledgeGraphAgent()
        except Exception as e:
            print(f"[KG] 标准Agent初始化失败，将使用降级模式：{e}")
            self.subject_name = "习近平新时代中国特色社会主义思想概论"
            self._llm = _KGLLM(model="qwen-max", temperature=0.5)
            self._graph_prompt = PromptTemplate.from_template(
                """
你是一位{subject_name}知识图谱专家。请围绕知识点"{topic}"构建一个 Mermaid mindmap（思维导图）格式的知识图谱，突出关键概念及其主要关系，并保持简洁易读。

输出要求：
1. mindmap 总节点不超过 15 个，层级不超过 3 级，保证图谱信息清晰、结构美观。
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
"""
            )

    def build_knowledge_graph(self, topic: str) -> str:
        if self._agent is not None:
            return self._agent.build_knowledge_graph(topic)
        prompt_text = self._graph_prompt.format(subject_name=self.subject_name, topic=topic)
        messages = [SystemMessage(content="你是一位精通知识图谱构建的学者。"), HumanMessage(content=prompt_text)]
        response = self._llm.invoke(messages)
        return str(getattr(response, "content", response)).strip()
    def _extract_topic(self, user_input: str) -> str:
        trigger_keywords = [
            "知识图谱", "思维导图", "mindmap", "图谱", "生成", "制作", "构建", "画", "帮我", "请", "关于", "：", ":",
        ]
        topic = user_input
        for kw in trigger_keywords:
            topic = topic.replace(kw, "")
        topic = topic.strip().lstrip("，,。 、")
        return topic if topic else user_input

    def process_request(self, user_input: str) -> str:
        topic = self._extract_topic(user_input)
        return self.build_knowledge_graph(topic)


app = Flask(__name__)

# 预先加载 .env，确保 Agent 初始化阶段可获取到密钥等环境变量
try:
    load_dotenv()
except Exception:
    pass

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
    kg_agent = KGAgentWrapper()
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
                    response_text = kg_agent.process_request(user_message)
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
                exam_keywords = ["出题", "生成题目", "选择题", "判断题", "简答题", "试题", "练习"]
                if any(kw in user_message for kw in exam_keywords):
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
                else:
                    if qa_agent:
                        apply_mode(qa_agent)
                        if image_path and hasattr(qa_agent, 'process_multimodal_request'):
                            response_text = qa_agent.process_multimodal_request(user_message, image_path)
                        else:
                            response_text = qa_agent.process_request(user_message)
                    else:
                        response_text = "问答助手未成功加载，无法处理您的请求。"
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



