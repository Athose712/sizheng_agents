# 思想道德与法治 智能助手（无角色扮演）

功能：问答（图片可选）、智能出题（按需显示解析）、知识图谱（Mermaid）。

## 快速开始
1) 安装依赖
```
pip install -r requirements.txt
```
2) 配置环境变量（Windows PowerShell）
```
$env:DASHSCOPE_API_KEY='your_api_key_here'
```
3) 准备知识库
- 将 PDF 资料放入 `sdfz_raw_data/`
- 构建：
```
python generate_database.py
```
4) 启动服务
```
python app.py
```
访问：`http://localhost:5021`

## 目录
- `app.py`：后端服务（仅聊天模式）
- `sixiangdaodefazhi_*.py`：三类 Agent（出题、知识图谱、问答）
- `templates/`、`static/`：前端页面与资源
- `sdfz_raw_data/`：原始 PDF 资料
- `database_agent_sixiangdaodefazhi/`：向量库


