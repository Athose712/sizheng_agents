// Reuse the same client logic as the Marxism app (no role-play parts)
document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const resetBtn = document.getElementById("reset-btn");
    const loading = document.getElementById("loading");

    const imageBtn = document.getElementById("image-btn");
    const imageInput = document.getElementById("image-input");
    const imagePreview = document.getElementById("image-preview");
    const previewImg = document.getElementById("preview-img");
    const removeImageBtn = document.getElementById("remove-image");

    let selectedImageData = null;

    try {
        if (window.mermaidMindmap && typeof mermaid.registerExternalDiagrams === 'function') {
            mermaid.registerExternalDiagrams([window.mermaidMindmap]);
        }
    } catch (e) { console.warn('Mermaid mindmap plugin registration skipped:', e); }
    mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' });

    const initialChatHTML = chatBox.innerHTML;
    const resetChat = () => {
        chatBox.innerHTML = initialChatHTML;
        userInput.value = "";
        clearSelectedImage();
    };

    const clearSelectedImage = () => {
        selectedImageData = null;
        imagePreview.style.display = 'none';
        previewImg.src = '';
        imageInput.value = '';
    };

    const handleImageSelection = (file) => {
        if (!file) return;
        if (!file.type.startsWith('image/')) { alert('请选择图片文件！'); return; }
        if (file.size > 16 * 1024 * 1024) { alert('图片文件过大，请选择小于16MB的图片！'); return; }
        const reader = new FileReader();
        reader.onload = (e) => {
            selectedImageData = e.target.result;
            previewImg.src = selectedImageData;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    };

    const showLoading = (show) => {
        loading.style.display = show ? 'block' : 'none';
        sendBtn.disabled = show;
    };

    const sendMessage = async () => {
        const query = userInput.value.trim();
        if (!query && !selectedImageData) { alert("请输入文本或选择图片！"); return; }
        appendMessage(query, "user", selectedImageData);

        const requestData = { message: query };
        const modeEl = document.getElementById('response-mode');
        if (modeEl && modeEl.value) { requestData.response_mode = modeEl.value; }
        if (selectedImageData) { requestData.image = selectedImageData; }

        userInput.value = "";
        clearSelectedImage();
        showLoading(true);

        try {
            const base = (window.__APP_BASE__ || "");
            const response = await fetch(`${base}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestData),
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            appendMessage(data.response, "bot");
        } catch (error) {
            console.error("Error:", error);
            appendMessage("抱歉，处理您的请求时出错，请查看控制台了解详情。", "bot");
        } finally { showLoading(false); }
    };

    const appendMessage = (content, type, imageData = null) => {
        const messageWrapper = document.createElement("div");
        messageWrapper.className = "message";
        const messageBubble = document.createElement("div");
        messageBubble.className = `message-bubble ${type === 'user' ? 'user-message' : 'bot-message'}`;

        if (type === 'user' && imageData) {
            const imageElement = document.createElement('img');
            imageElement.src = imageData;
            imageElement.style.maxWidth = '200px';
            imageElement.style.maxHeight = '150px';
            imageElement.style.borderRadius = '10px';
            imageElement.style.marginBottom = '10px';
            imageElement.style.display = 'block';
            messageBubble.appendChild(imageElement);
        }

        const mermaidBlockRegex = /```mermaid([\s\S]*?)```/i;
        const match = content && content.match ? content.match(mermaidBlockRegex) : null;

        if (type === 'bot' && match) {
            const mermaidSource = match[1].trim();
            const summaryText = content.replace(match[0], '').trim();
            const graphDiv = document.createElement('div');
            graphDiv.className = 'mermaid';
            graphDiv.textContent = mermaidSource;
            messageBubble.appendChild(graphDiv);
            if (summaryText) {
                const summaryDiv = document.createElement('div');
                summaryDiv.className = 'mermaid-summary';
                summaryDiv.innerHTML = marked.parse(summaryText);
                messageBubble.appendChild(summaryDiv);
            }
            setTimeout(() => {
                try {
                    if (typeof mermaid.registerExternalDiagrams === 'function') {
                        const mindmapPlugin = window.mermaidMindmap || window.mindmap;
                        if (mindmapPlugin) { try { mermaid.registerExternalDiagrams([mindmapPlugin]); } catch (_) {} }
                    }
                    mermaid.init(undefined, graphDiv);
                } catch (e) {
                    console.error('Mermaid render error:', e);
                    graphDiv.innerHTML = `<pre><code>${mermaidSource}</code></pre>`;
                }
            }, 100);
        } else if (type === 'bot') {
            const maybeQuiz = tryParseQuiz(content || '');
            if (maybeQuiz && maybeQuiz.length > 0) {
                const quizContainer = renderQuizCards(maybeQuiz);
                messageBubble.appendChild(quizContainer);
            } else {
                messageBubble.innerHTML = marked.parse(content || '');
            }
        } else {
            if (content) {
                const textDiv = document.createElement('div');
                textDiv.textContent = content;
                messageBubble.appendChild(textDiv);
            }
        }

        messageWrapper.appendChild(messageBubble);
        chatBox.appendChild(messageWrapper);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    function tryParseQuiz(text) {
        const indicatorRegex = /(题目\s*\d+|选择题\s*\d+|判断题\s*\d+|简答题\s*\d+|题干\s*[:：]|正确答案\s*[:：]|参考答案\s*[:：]|解析\s*[:：])/;
        if (!indicatorRegex.test(text)) return null;
        const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
        if (lines.length === 0) return null;
        const titleRegex = /^(?:题目|选择题|判断题|简答题)\s*\d+\s*[:：]?/;
        const indices = [];
        for (let i = 0; i < lines.length; i++) { if (titleRegex.test(lines[i])) indices.push(i); }
        if (indices.length === 0) { indices.push(0); }
        indices.push(lines.length);
        const questions = [];
        for (let k = 0; k < indices.length - 1; k++) {
            const start = indices[k];
            const end = indices[k + 1];
            const chunk = lines.slice(start, end);
            if (chunk.length === 0) continue;
            let title = chunk[0].replace(/\s+/g, '');
            let stem = '';
            const options = [];
            let answer = '';
            let explanation = '';
            const stemRegex = /^题干\s*[:：]\s*(.*)$/;
            const optionRegex = /^[A-DＡ-Ｄ]\s*[\.．、]\s*(.*)$/;
            const answerRegex = /^(?:正确?答案|参考答案)\s*[:：]\s*(.*)$/;
            const explainRegex = /^(?:解析|答案解析|解答|讲解)\s*[:：]\s*(.*)$/;
            for (let i = 0; i < chunk.length; i++) {
                const line = chunk[i];
                if (i === 0 && titleRegex.test(line)) continue;
                let m;
                if ((m = line.match(stemRegex))) { stem = m[1]; continue; }
                if ((m = line.match(optionRegex))) {
                    const prefixMatch = line.match(/^[A-DＡ-Ｄ]/);
                    const prefix = prefixMatch ? prefixMatch[0].replace(/[Ａ-Ｄ]/, c => String.fromCharCode(c.charCodeAt(0) - 65248)) : '';
                    const text = line.replace(/^[A-DＡ-Ｄ]\s*[\.．、]\s*/, '');
                    options.push(`${prefix}. ${text}`);
                    continue;
                }
                if ((m = line.match(answerRegex))) { answer = m[1]; continue; }
                if ((m = line.match(explainRegex))) {
                    const parts = [m[1]];
                    for (let j = i + 1; j < chunk.length; j++) {
                        const nxt = chunk[j];
                        if (stemRegex.test(nxt) || optionRegex.test(nxt) || answerRegex.test(nxt) || titleRegex.test(nxt)) break;
                        parts.push(nxt); i = j;
                    }
                    explanation = parts.join('\n');
                    continue;
                }
            }
            if (!stem) { stem = chunk.slice(1).find(l => !optionRegex.test(l) && !answerRegex.test(l) && !explainRegex.test(l)) || ''; }
            questions.push({ title, stem, options, answer, explanation });
        }
        return questions;
    }

    function renderQuizCards(questions) {
        const container = document.createElement('div');
        container.className = 'quiz-list';
        questions.forEach((q, idx) => {
            const card = document.createElement('div');
            card.className = 'quiz-card';
            const header = document.createElement('div'); header.className = 'quiz-header'; header.textContent = q.title || `题目${idx + 1}`; card.appendChild(header);
            if (q.stem) { const stem = document.createElement('div'); stem.className = 'quiz-stem'; stem.textContent = q.stem; card.appendChild(stem); }
            if (q.options && q.options.length > 0) {
                const optWrap = document.createElement('ul'); optWrap.className = 'quiz-options';
                q.options.forEach(op => { const li = document.createElement('li'); li.textContent = op; optWrap.appendChild(li); });
                card.appendChild(optWrap);
            }
            const hasAnswer = q.answer && q.answer.trim().length > 0;
            const hasExplain = q.explanation && q.explanation.trim().length > 0;
            if (hasAnswer || hasExplain) {
                const actionBar = document.createElement('div'); actionBar.className = 'quiz-actions';
                const toggleBtn = document.createElement('button'); toggleBtn.className = 'quiz-toggle-btn'; toggleBtn.textContent = '显示解析';
                actionBar.appendChild(toggleBtn); card.appendChild(actionBar);
                const detail = document.createElement('div'); detail.className = 'quiz-detail hidden';
                if (hasAnswer) { const ans = document.createElement('div'); ans.className = 'quiz-answer'; ans.innerHTML = `<span class="label">正确答案</span><span class="value">${escapeHTML(q.answer)}</span>`; detail.appendChild(ans); }
                if (hasExplain) { const exp = document.createElement('div'); exp.className = 'quiz-explain'; exp.innerHTML = `<span class="label">解析</span><div class="value">${escapeHTML(q.explanation).replace(/\n/g,'<br>')}</div>`; detail.appendChild(exp); }
                card.appendChild(detail);
                toggleBtn.addEventListener('click', () => {
                    const isHidden = detail.classList.contains('hidden');
                    detail.classList.toggle('hidden');
                    toggleBtn.textContent = isHidden ? '隐藏解析' : '显示解析';
                });
            }
            container.appendChild(card);
        });
        return container;
    }

    function escapeHTML(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    sendBtn.addEventListener("click", sendMessage);
    resetBtn.addEventListener("click", resetChat);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    imageBtn.addEventListener("click", () => { imageInput.click(); });
    imageInput.addEventListener("change", (e) => { const file = e.target.files[0]; if (file) handleImageSelection(file); });
    removeImageBtn.addEventListener("click", clearSelectedImage);
    document.addEventListener("dragover", (e) => { e.preventDefault(); });
    document.addEventListener("drop", (e) => { e.preventDefault(); const files = e.dataTransfer.files; if (files.length > 0) handleImageSelection(files[0]); });
    document.addEventListener("paste", (e) => {
        const items = (e.clipboardData || window.clipboardData).items; if (!items) return;
        for (let i = 0; i < items.length; i++) { const item = items[i]; if (item && item.type && item.type.startsWith('image/')) { const file = item.getAsFile(); if (file) { handleImageSelection(file); e.preventDefault(); break; } } }
    });
});


