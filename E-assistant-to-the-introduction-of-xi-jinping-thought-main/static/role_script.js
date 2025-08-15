document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('role-form');
  if (!form) return;
  const messageEl = document.getElementById('message');
  const imageEl = document.getElementById('image');
  const modeEl = document.getElementById('response_mode');
  const dialogueEl = document.getElementById('dialogue');
  let sessionId = null;

  async function toBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = (messageEl.value || '').trim();
    const mode = modeEl.value || 'balanced';
    let image = null;
    if (imageEl.files && imageEl.files[0]) {
      image = await toBase64(imageEl.files[0]);
    }
    dialogueEl.textContent = '思考中...';
    try {
      const url = sessionId ? 'continue_dialogue' : 'start_dialogue';
      const payload = { response_mode: mode };
      if (message) payload.message = message;
      if (image) payload.image = image;
      if (sessionId) payload.session_id = sessionId;
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (data.error) {
        dialogueEl.textContent = data.error;
        return;
      }
      if (data.session_id) sessionId = data.session_id;
      const prefix = data.character && data.topic ? `[${data.character}｜${data.topic}]` : '';
      dialogueEl.textContent = `${prefix}\n${data.response || '无响应'}`;
      messageEl.value = '';
      imageEl.value = '';
    } catch (err) {
      dialogueEl.textContent = '请求失败';
    }
  });
});


