import os
import tempfile
import base64
from io import BytesIO
from typing import Optional

from flask import Flask, request, jsonify, render_template
from PIL import Image

from sixiangdaodefazhi_agent import SixiangDaodeFazhiQuestionAgent
from sixiangdaodefazhi_kg_agent import SixiangDaodeFazhiKnowledgeGraphAgent
from sixiangdaodefazhi_qa_agent import SixiangDaodeFazhiAnswerAgent
from dotenv import load_dotenv


class KGAgentWrapper(SixiangDaodeFazhiKnowledgeGraphAgent):
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

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', tempfile.gettempdir())
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
            print("图片文件过大：超过16MB限制")
            return None

        try:
            img = Image.open(BytesIO(image_binary))
            img.verify()
            real_format = img.format
            if img.width > 4096 or img.height > 4096:
                print("图片分辨率过高：超过 4K 限制，拒绝处理")
                return None
        except Exception as e:
            print(f"无效的图像文件: {e}")
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
    except Exception as e:
        print(f"保存图片失败: {e}")
        return None


def cleanup_temp_file(file_path: Optional[str]) -> None:
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"清理临时文件失败: {e}")


try:
    question_agent = SixiangDaodeFazhiQuestionAgent()
    print("SixiangDaodeFazhiQuestionAgent loaded successfully.")
except Exception as e:
    print(f"Error loading SixiangDaodeFazhiQuestionAgent: {e}")
    question_agent = None

try:
    kg_agent = KGAgentWrapper()
    print("KGAgentWrapper loaded successfully.")
except Exception as e:
    print(f"Error loading KGAgentWrapper: {e}")
    kg_agent = None

try:
    qa_agent = SixiangDaodeFazhiAnswerAgent()
    print("SixiangDaodeFazhiAnswerAgent loaded successfully.")
except Exception as e:
    print(f"Error loading SixiangDaodeFazhiAnswerAgent: {e}")
    qa_agent = None


@app.route('/chat_ui')
def chat_ui():
    return render_template('index.html')


@app.route('/')
def home():
    return render_template('home.html')


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
                print("Routing to Knowledge Graph Agent.")
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
                if re.search(r"[A-DＡ-Ｄ][\.．、]\\s?", user_message):
                    contains_mcq_options = True
            except Exception:
                contains_mcq_options = False

            if any(kw in user_message for kw in answer_keywords) or contains_mcq_options:
                if qa_agent:
                    print("Routing to Q&A Agent (answer mode).")
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
                        print("Routing to Question Generation Agent.")
                        apply_mode(question_agent)
                        if hasattr(question_agent, 'process_multimodal_request'):
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
                        print("Routing to Q&A Agent (default).")
                        apply_mode(qa_agent)
                        if image_path and hasattr(qa_agent, 'process_multimodal_request'):
                            response_text = qa_agent.process_multimodal_request(user_message, image_path)
                        else:
                            response_text = qa_agent.process_request(user_message)
                    else:
                        response_text = "问答助手未成功加载，无法处理您的请求。"
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        response_text = f"处理您的请求时发生内部错误: {e}"
    finally:
        if image_path:
            cleanup_temp_file(image_path)

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

    app.run(debug=True, port=5021)


if __name__ == '__main__':
    run_app()


