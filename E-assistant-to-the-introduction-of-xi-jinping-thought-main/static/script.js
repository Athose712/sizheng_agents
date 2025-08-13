document.addEventListener('DOMContentLoaded', () => {
  const chatBox = document.getElementById('chat-box');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const resetBtn = document.getElementById('reset-btn');
  const endBtn = document.getElementById('end-btn');
  const loading = document.getElementById('loading');
  const imageBtn = document.getElementById('image-btn');
  const imageInput = document.getElementById('image-input');
  const imagePreview = document.getElementById('image-preview');
  const previewImg = document.getElementById('preview-img');
  const removeImageBtn = document.getElementById('remove-image');
  const modeSelect = document.getElementById('response-mode');

  let selectedImageData = null;

  try {
    if (window.mermaidMindmap && typeof mermaid.registerExternalDiagrams === 'function') {
      mermaid.registerExternalDiagrams([window.mermaidMindmap]);
    }
  } catch (e) {}
  if (window.mermaid) mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' });

  const initialChatHTML = chatBox.innerHTML;

  const resetChat = () => {
    chatBox.innerHTML = initialChatHTML;
    userInput.value = '';
    clearSelectedImage();
    if (endBtn) endBtn.style.display = 'none';
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
    reader.onload = (e) => { selectedImageData = e.target.result; previewImg.src = selectedImageData; imagePreview.style.display = 'block'; };
    reader.readAsDataURL(file);
  };

  const showLoading = (show) => { loading.style.display = show ? 'block' : 'none'; sendBtn.disabled = show; };

  const sendMessage = async () => {
    const text = userInput.value.trim();
    if (!text && !selectedImageData) return;
    appendMessage(text || '（仅图片）', 'user', selectedImageData);
    const payload = { message: text };
    if (modeSelect && modeSelect.value) payload.response_mode = modeSelect.value;
    if (selectedImageData) payload.image = selectedImageData;
    userInput.value = '';
    clearSelectedImage();
    showLoading(true);
    try {
      const resp = await fetch('./chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = await resp.json();
      appendMessage(data.response || data.error || '（无回复）', 'bot');
    } catch (e) {
      appendMessage('网络错误，请稍后重试。', 'bot');
    } finally { showLoading(false); }
  };

  const appendMessage = (content, type, imageData = null) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'message';
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${type === 'user' ? 'user-message' : 'bot-message'}`;
    if (type === 'user' && imageData) {
      const img = document.createElement('img');
      img.src = imageData; img.style.maxWidth = '200px'; img.style.maxHeight = '150px'; img.style.borderRadius = '10px'; img.style.marginBottom = '10px'; img.style.display = 'block';
      bubble.appendChild(img);
    }
    if (type === 'bot') {
      const mermaidBlock = content.match(/```mermaid([\s\S]*?)```/i);
      if (mermaidBlock) {
        const mermaidSrc = mermaidBlock[1].trim();
        const summary = content.replace(mermaidBlock[0], '').trim();
        const graphDiv = document.createElement('div'); graphDiv.className = 'mermaid'; graphDiv.textContent = mermaidSrc; bubble.appendChild(graphDiv);
        if (summary) { const s = document.createElement('div'); s.innerHTML = marked.parse(summary); bubble.appendChild(s); }
        setTimeout(() => { try { if (window.mermaid) mermaid.init(undefined, graphDiv); } catch(e) { graphDiv.innerHTML = `<pre><code>${mermaidSrc}</code></pre>`; } }, 50);
      } else {
        bubble.innerHTML = marked.parse(content);
      }
    } else {
      if (content) { const t = document.createElement('div'); t.textContent = content; bubble.appendChild(t); }
    }
    wrapper.appendChild(bubble); chatBox.appendChild(wrapper); chatBox.scrollTop = chatBox.scrollHeight;
  };

  sendBtn.addEventListener('click', sendMessage);
  resetBtn.addEventListener('click', resetChat);
  if (endBtn) endBtn.addEventListener('click', resetChat);
  userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
  imageBtn.addEventListener('click', () => imageInput.click());
  imageInput.addEventListener('change', (e) => { const f = e.target.files[0]; if (f) handleImageSelection(f); });
  removeImageBtn.addEventListener('click', clearSelectedImage);
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => { e.preventDefault(); const files = e.dataTransfer.files; if (files.length > 0) handleImageSelection(files[0]); });
  document.addEventListener('paste', (e) => { const items = (e.clipboardData || window.clipboardData).items; if (!items) return; for (let i = 0; i < items.length; i++) { const item = items[i]; if (item && item.type && item.type.startsWith('image/')) { const file = item.getAsFile(); if (file) { handleImageSelection(file); e.preventDefault(); break; } } } });
});


