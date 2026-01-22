// app/ui/static/js/main.js

document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const inputField = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const message = inputField.value.trim();

    if (!message) return;

    // 1. Añadir mensaje del usuario a la UI
    appendMessage('user', message);
    inputField.value = '';
    inputField.disabled = true;
    sendBtn.classList.add('is-loading');

    try {
        // 2. Llamada a tu API FastAPI
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: "cachito_2",
                message: message,
                max_tokens: 150,
                temperature: 0.8
            })
        });

        const data = await response.json();

        if (response.ok) {
            // 3. Añadir respuesta del modelo
            appendMessage('assistant', data.response);
            document.getElementById('token-info').innerText = 
                `Tokens: ${data.prompt_tokens} (prompt) + ${data.response_tokens} (gen) = ${data.total_tokens}`;
        } else {
            appendMessage('assistant', "Error: No pude procesar tu solicitud.");
        }
    } catch (error) {
        console.error(error);
        appendMessage('assistant', "Error de conexión con el servidor.");
    } finally {
        inputField.disabled = false;
        sendBtn.classList.remove('is-loading');
        inputField.focus();
    }
});

function appendMessage(role, text) {
    const chatContainer = document.getElementById('chat-container');
    const msgDiv = document.createElement('div');
    msgDiv.className = `notification ${role === 'user' ? 'is-info is-light has-text-right' : 'is-success is-light'} mb-2`;
    
    // Escapar HTML básico para seguridad
    msgDiv.innerHTML = `<strong>${role === 'user' ? 'Tú' : 'Cachito'}</strong><br>${text}`;
    
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight; // Auto-scroll
}