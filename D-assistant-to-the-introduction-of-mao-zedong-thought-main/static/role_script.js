document.addEventListener('DOMContentLoaded', () => {
	const chatBox = document.getElementById('chat-box');
	const userInput = document.getElementById('user-input');
	const sendBtn = document.getElementById('send-btn');
	const endBtn = document.getElementById('end-btn');
	const loading = document.getElementById('loading');
	const imageBtn = document.getElementById('image-btn');
	const imageInput = document.getElementById('image-input');
	const imagePreview = document.getElementById('image-preview');
	const previewImg = document.getElementById('preview-img');
	const removeImage = document.getElementById('remove-image');
	const modeSelect = document.getElementById('response-mode');

	let sessionId = null;
	let imageBase64 = null;

	const showLoading = (show) => {
		loading.style.display = show ? 'block' : 'none';
		sendBtn.disabled = show;
		if (endBtn) endBtn.disabled = show;
	};

	const appendMessage = (content, type) => {
		const wrapper = document.createElement('div');
		wrapper.className = 'message';
		const bubble = document.createElement('div');
		bubble.className = `message-bubble ${type === 'user' ? 'user-message' : 'bot-message'}`;
		bubble.textContent = content;
		wrapper.appendChild(bubble);
		chatBox.appendChild(wrapper);
		chatBox.scrollTop = chatBox.scrollHeight;
	};

	const resetChat = () => {
		chatBox.innerHTML = '';
		sessionId = null;
		userInput.value = '';
		imageBase64 = null;
		imagePreview.style.display = 'none';
		previewImg.src = '';
		imageInput.value = '';
	};

	const endDialogue = async () => {
		if (!sessionId) {
			resetChat();
			return;
		}
		showLoading(true);
		try {
			await fetch('end_dialogue', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ session_id: sessionId }),
			});
		} catch (err) {
			console.error(err);
		} finally {
			showLoading(false);
			resetChat();
			appendMessage('（对话已结束，您可以开始新的对话）', 'bot');
		}
	};

	const sendMessage = async () => {
		const text = (userInput.value || '').trim();
		if (!text && !imageBase64) return;
		appendMessage(text || '（仅图片）', 'user');
		userInput.value = '';
		showLoading(true);

		const url = sessionId ? 'continue_dialogue' : 'start_dialogue';
		const payload = sessionId ? { session_id: sessionId, message: text } : { message: text };
		if (modeSelect && modeSelect.value) payload.response_mode = modeSelect.value;
		if (imageBase64) payload.image = imageBase64;

		try {
			const response = await fetch(url, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload),
			});
			const data = await response.json();
			if (response.ok) {
				if (data.session_id) sessionId = data.session_id;
				appendMessage(data.response || data.message || '（无回复）', 'bot');
			} else {
				appendMessage(data.error || '发生错误，请稍后重试。', 'bot');
			}
		} catch (err) {
			console.error(err);
			appendMessage('网络错误，请检查连接。', 'bot');
		} finally {
			showLoading(false);
			imageBase64 = null;
			imagePreview.style.display = 'none';
			previewImg.src = '';
			imageInput.value = '';
		}
	};

	imageBtn.addEventListener('click', () => imageInput.click());
	imageInput.addEventListener('change', () => {
		const file = imageInput.files && imageInput.files[0];
		if (!file) return;
		const reader = new FileReader();
		reader.onload = () => {
			imageBase64 = reader.result;
			imagePreview.style.display = 'block';
			previewImg.src = imageBase64;
		};
		reader.readAsDataURL(file);
	});
	removeImage.addEventListener('click', () => {
		imageBase64 = null;
		imagePreview.style.display = 'none';
		previewImg.src = '';
		imageInput.value = '';
	});

	sendBtn.addEventListener('click', sendMessage);
	endBtn.addEventListener('click', endDialogue);
	userInput.addEventListener('keypress', (e) => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
	});
});


