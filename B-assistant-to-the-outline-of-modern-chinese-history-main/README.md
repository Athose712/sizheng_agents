# 中国近现代史纲要 智能助手（无角色扮演）

本项目提供与“马原”项目相同的核心能力：
- 问答（支持图片辅助）
- 智能出题（按需显示解析）
- 知识图谱（Mermaid 思维导图）

## 快速开始

1) 安装依赖
```
pip install -r requirements.txt
```

2) 配置环境变量（以 Windows PowerShell 为例）
```
$env:DASHSCOPE_API_KEY='your_api_key_here'
```

3) 准备知识库
- 将 PDF 资料放入 `jindaishi_raw_data/`
- 运行：
```
python generate_database.py
```

4) 启动服务
```
python app.py
```
访问：`http://localhost:5011`

## 目录结构
- `app.py`：后端服务（仅聊天模式）
- `jindaishi_*.py`：三类 Agent（出题、知识图谱、问答）
- `templates/`、`static/`：前端页面与资源
- `jindaishi_raw_data/`：原始 PDF 资料
- `database_agent_jindaishi/`：向量库


