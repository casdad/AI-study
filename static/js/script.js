// 初始化 Socket.IO 客户端连接
const socket = io();

// 获取 DOM 元素
const videoSourceSelect = document.getElementById('videoSource');
const filePathContainer = document.getElementById('filePathContainer');
const videoFilePathInput = document.getElementById('videoFilePath');
const browseFileBtn = document.getElementById('browseFileBtn');
const hiddenFileInput = document.getElementById('hiddenFileInput');
const fileInfoLabel = document.getElementById('fileInfoLabel');
const startStopBtn = document.getElementById('startStopBtn');
const analyzeFrameBtn = document.getElementById('analyzeFrameBtn');
const resetAppBtn = document.getElementById('resetAppBtn');

const objectList = document.getElementById('objectList');
const zhipuInfoText = document.getElementById('zhipuInfoText');
const chatHistoryText = document.getElementById('chatHistoryText');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');
const chatStatusLabel = document.getElementById('chatStatusLabel');
const videoFeedImg = document.getElementById('videoFeed'); // 获取视频流的图像元素

let isVideoRunning = false; // 标记视频流是否正在运行
let selectedFile = null; // 存储用户选择的文件对象

// --- UI 状态更新函数 ---
function updateUIForSource() {
    // 根据视频源选择更新 UI
    if (videoSourceSelect.value === 'webcam') {
        filePathContainer.style.display = 'none';
        fileInfoLabel.innerText = '使用默认摄像头 (索引 0)';
        selectedFile = null; // 清除文件选择
        videoFilePathInput.value = ''; // 清除文件路径输入框
    } else {
        filePathContainer.style.display = 'block';
        if (selectedFile && selectedFile.serverPath) { // 检查上传后服务器路径是否存在
            videoFilePathInput.value = selectedFile.name;
            fileInfoLabel.innerText = `已选择文件: ${selectedFile.name}`;
        } else {
            videoFilePathInput.value = '';
            fileInfoLabel.innerText = '请选择视频文件';
        }
    }
    // 确保浏览按钮状态正确
    browseFileBtn.disabled = isVideoRunning || videoSourceSelect.value === 'webcam';
}

function setRunningState(running) {
    // 更新按钮和下拉框的状态
    isVideoRunning = running;
    startStopBtn.innerText = running ? '停止识别' : '开始识别';
    startStopBtn.classList.toggle('btn-danger', running); // 运行中为红色，未运行为蓝色
    startStopBtn.classList.toggle('btn-primary', !running);

    analyzeFrameBtn.disabled = running; // 视频流运行时禁用“识别当前帧”按钮
    videoSourceSelect.disabled = running;
    browseFileBtn.disabled = running || videoSourceSelect.value === 'webcam';
    videoFilePathInput.disabled = running || videoSourceSelect.value === 'webcam';
}

// --- 事件监听器 ---
videoSourceSelect.addEventListener('change', updateUIForSource);

browseFileBtn.addEventListener('click', () => {
    hiddenFileInput.click(); // 模拟点击隐藏的文件输入框
});

hiddenFileInput.addEventListener('change', async (event) => {
    // 处理文件选择
    const file = event.target.files[0];
    if (file) {
        selectedFile = file; // 存储文件对象
        videoFilePathInput.value = file.name;
        fileInfoLabel.innerText = `正在上传文件: ${file.name}...`;

        const formData = new FormData();
        formData.append('video_file', file);

        try {
            const uploadResponse = await fetch('/upload_video', { method: 'POST', body: formData });
            const uploadData = await uploadResponse.json();
            if (uploadData.status === 'success') {
                selectedFile.serverPath = uploadData.filepath; // 存储服务器返回的路径
                fileInfoLabel.innerText = `文件上传成功: ${selectedFile.name}`;
                console.log('文件上传成功:', selectedFile.serverPath);
            } else {
                alert('文件上传失败: ' + uploadData.message);
                selectedFile = null; // 上传失败时清除文件选择
                videoFilePathInput.value = '';
                fileInfoLabel.innerText = '文件上传失败，请重新选择或检查文件类型。';
            }
        } catch (error) {
            console.error('上传时发生错误:', error);
            alert('文件上传时发生网络错误。');
            selectedFile = null;
            videoFilePathInput.value = '';
            fileInfoLabel.innerText = '文件上传时发生网络错误。';
        }
    } else {
        selectedFile = null;
        videoFilePathInput.value = '';
        fileInfoLabel.innerText = '请选择视频文件';
    }
});

startStopBtn.addEventListener('click', async () => {
    if (isVideoRunning) {
        // 如果正在运行，发送停止请求
        const response = await fetch('/start_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 'action': 'stop' }) // 发送停止信号
        });
        const data = await response.json();
        console.log(data.message);
        setRunningState(false);
    } else {
        // 如果未运行，发送开始请求
        const sourceType = videoSourceSelect.value;
        let path = null;
        if (sourceType === 'file') {
            if (!selectedFile || !selectedFile.serverPath) {
                alert('请先选择一个有效的视频文件并等待上传完成！');
                return;
            }
            path = selectedFile.serverPath; // 使用服务器返回的文件路径
        }

        const response = await fetch('/start_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_type: sourceType, path: path })
        });
        const data = await response.json();
        if (data.status === 'error') {
            alert('启动失败: ' + data.message);
        } else {
            setRunningState(true);
            console.log(data.message);
            // 添加缓存清除参数，确保视频流能重新加载
            videoFeedImg.src = "/video_feed?" + new Date().getTime(); // 直接使用硬编码路径，因为这个JS文件不是Jinja模板
        }
    }
});

analyzeFrameBtn.addEventListener('click', async () => {
    const response = await fetch('/analyze_frame', { method: 'POST' });
    const data = await response.json();
    if (data.status === 'error') {
        alert('分析失败: ' + data.message);
    } else {
        console.log(data.message);
        // 单帧分析后，视频流会停止，更新按钮状态
        setRunningState(false);
    }
});

resetAppBtn.addEventListener('click', async () => {
    const response = await fetch('/reset_app', { method: 'POST' });
    const data = await response.json();
    console.log(data.message);
    setRunningState(false);
    // 本地清理显示
    objectList.innerHTML = '';
    zhipuInfoText.innerHTML = '';
    chatHistoryText.innerHTML = ''; // SocketIO 连接后会重新添加系统消息
    chatStatusLabel.innerText = '';
    videoSourceSelect.value = 'webcam'; // 重置视频源选择
    selectedFile = null;
    videoFilePathInput.value = ''; // 清空文件路径输入框
    videoFeedImg.src = "/video_feed"; // 重置视频流 src，直接使用硬编码路径
    updateUIForSource(); // 更新文件路径显示和按钮状态
});

// --- 聊天机器人功能 ---
chatSendBtn.addEventListener('click', () => sendChatMessage());
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendChatMessage();
    }
});

function sendChatMessage() {
    const message = chatInput.value.trim();
    if (message) {
        socket.emit('send_chat_message', { message: message }); // 通过 SocketIO 发送消息
        chatInput.value = '';
        chatSendBtn.disabled = true; // 发送期间禁用输入
        chatInput.disabled = true;
        chatStatusLabel.innerText = "AI: 正在思考...";
        chatStatusLabel.style.color = "blue";
    }
}

function displayChatMessage(message, senderType = 'ai') {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add(senderType === 'user' ? 'user-message' : (senderType === 'system' ? 'system-message' : 'ai-message'));
    msgDiv.innerHTML = message.replace(/\n/g, '<br>'); // 处理换行符
    chatHistoryText.appendChild(msgDiv);
    chatHistoryText.scrollTop = chatHistoryText.scrollHeight; // 滚动到底部
}

// --- SocketIO 事件处理函数 (接收服务器消息) ---
socket.on('detected_objects_update', (data) => {
    objectList.innerHTML = '';
    if (data.objects && data.objects.length > 0) {
        data.objects.forEach(obj => {
            const li = document.createElement('li');
            li.innerText = obj;
            objectList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.innerText = '未检测到物体';
        li.style.color = '#888';
        objectList.appendChild(li);
    }
});

socket.on('zhipu_info_update', (data) => {
    if (data.clear) {
        zhipuInfoText.innerHTML = '';
    } else if (data.info) {
        const p = document.createElement('p');
        p.innerHTML = data.info.replace(/\n/g, '<br>');
        zhipuInfoText.appendChild(p);
        zhipuInfoText.scrollTop = zhipuInfoText.scrollHeight;
    }
});

socket.on('chat_response', (data) => {
    if (data.user_message) {
        displayChatMessage(`你: ${data.user_message}`, 'user');
    } else if (data.is_system) { // 系统消息
        displayChatMessage(`AI: ${data.message}`, 'system');
    }
    else if (data.success) {
        displayChatMessage(`AI: ${data.message}`, 'ai');
    } else {
        displayChatMessage(`AI: 抱歉，发生错误：${data.message}`, 'ai');
    }
    chatSendBtn.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
});

socket.on('chat_status', (data) => {
    chatStatusLabel.innerText = data.message;
    chatStatusLabel.style.color = data.color;
});

socket.on('status_update', (data) => {
    console.log('服务器状态:', data.message);
    document.getElementById('statusMessage').innerText = data.message; // 更新视频卡片底部的状态消息
});

socket.on('clear_displays', () => {
    objectList.innerHTML = '';
    zhipuInfoText.innerHTML = '';
    chatHistoryText.innerHTML = ''; // 服务器端重置时会发送系统消息
    chatStatusLabel.innerText = '';
});

// 页面加载完成后的初始设置
document.addEventListener('DOMContentLoaded', () => {
    updateUIForSource();
    setRunningState(false);
    // 确保视频流图像的alt文本正确
    videoFeedImg.alt = '视频流正在加载...';
});
