const startBtn = document.getElementById('start-btn');
const welcomeScreen = document.getElementById('welcome-screen');
const chatInterface = document.getElementById('chat-interface');
const bgLayer = document.getElementById('bg-layer');
const bgOverlay = document.getElementById('bg-overlay');

const chatForm = document.getElementById('chat-form');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');

// chuyển cảnh
startBtn.addEventListener('click', () => {
    welcomeScreen.style.opacity = '0';
    welcomeScreen.style.transition = 'opacity 0.5s ease';
    setTimeout(() => welcomeScreen.classList.add('hidden'), 500);

    bgLayer.classList.add('bg-dimmed');
    
    chatInterface.classList.remove('hidden');
    setTimeout(() => {
        chatInterface.style.opacity = '1';
    }, 100);
});

// main logic
function createMessageHTML(text, sender) {
    const isUser = sender === 'user';
    
    if (isUser) {
        return `
        <div class="flex justify-end animate-fade-in-up">
            <div class="bg-teal-600 text-white p-3 px-5 rounded-2xl rounded-tr-none shadow-lg max-w-[85%]">
                ${text}
            </div>
        </div>`;
    } else {
        return `
        <div class="flex gap-4 animate-fade-in-up">
            <div class="w-8 h-8 rounded-full bg-teal-600 flex items-center justify-center flex-shrink-0 shadow-lg">
                <i class="fa-solid fa-robot text-sm"></i>
            </div>
            <div class="bg-white/10 border border-white/10 p-4 rounded-2xl rounded-tl-none text-gray-100 max-w-[85%] shadow-sm">
                ${text}
            </div>
        </div>`;
    }
}

function showLoading() {
    const id = "loading-" + Date.now();
    const html = `
        <div id="${id}" class="flex gap-4 animate-fade-in-up">
             <div class="w-8 h-8 rounded-full bg-teal-600 flex items-center justify-center flex-shrink-0">
                <i class="fa-solid fa-robot text-sm"></i>
            </div>
            <div class="bg-white/10 border border-white/10 p-4 rounded-2xl rounded-tl-none h-12 flex items-center gap-1">
                <div class="w-1.5 h-1.5 rounded-full typing-dot"></div>
                <div class="w-1.5 h-1.5 rounded-full typing-dot"></div>
                <div class="w-1.5 h-1.5 rounded-full typing-dot"></div>
            </div>
        </div>
    `;
    chatBox.insertAdjacentHTML('beforeend', html);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    // Hiển thị tin nhắn của người dùng
    chatBox.insertAdjacentHTML('beforeend', createMessageHTML(message, 'user'));
    chatBox.scrollTop = chatBox.scrollHeight;
    userInput.value = '';

    // Hiển thị loading indicator
    const loadingId = showLoading();

    try {
        const formData = new FormData();
        formData.append('prompt', message);

        const response = await fetch('/chat', { method: 'POST', body: formData });

        if (!response.ok) {
            throw new Error(`Server trả về lỗi: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let botMessage = ""; // Lưu trữ toàn bộ câu trả lời

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            botMessage += chunk;
            const loadingElement = document.getElementById(loadingId);
            if (loadingElement) {
                loadingElement.remove();
            }
            chatBox.insertAdjacentHTML('beforeend', createMessageHTML(chunk.trim(), 'bot'));
            chatBox.scrollTop = chatBox.scrollHeight;
        }

    } catch (error) {
        console.error("Lỗi khi gửi yêu cầu:", error);
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }
        chatBox.insertAdjacentHTML('beforeend', createMessageHTML("Lỗi kết nối! Vui lòng thử lại.", 'bot'));
    }
});