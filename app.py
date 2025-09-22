# app.py

# 确保 eventlet.monkey_patch() 在所有其他导入之前执行，以避免 Werkzeug 相关的上下文错误
try:
    import eventlet
    eventlet.monkey_patch()
except ImportError:
    print("警告: eventlet 未安装，为了更好的 SocketIO 性能，请考虑安装 eventlet 或 gevent。")

import os
import cv2
from PIL import Image
from ultralytics import YOLO
from zhipuai import ZhipuAI # 确保 ZhipuAI 库已安装
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import base64
import numpy as np # 用于处理图像数据

# Flask and SocketIO imports
from flask import Flask, render_template, request, Response, jsonify
from flask_socketio import SocketIO, emit

# For secure file uploads
from werkzeug.utils import secure_filename

# --- Configuration ---
API_KEY_WARNING_PLACEHOLDER = "YOUR_ZHIPU_API_KEY_HERE" # 占位符，如果API Key未配置
# 从环境变量获取 ZHIPU_API_KEY，如果没有设置，则使用提供的默认密钥（请务必替换为您的实际密钥！）
# 根据您提供的信息，您的真实密钥是 "80225b6e07c94c24b48240fe1086c239.fU82WkuhAgY51eRB"
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "80225b6e07c94c24b48240fe1086c239.fU82WkuhAgY51eRB")
YOLO_MODEL_PATH = 'yolov8n.pt' # 确保模型文件位于同一目录或指定正确路径
UPLOAD_FOLDER = 'static/uploads' # 网页可访问的上传文件目录
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'} # 允许上传的视频文件类型

# --- 添加英文到中文的物体名称映射字典 ---
# 您可以根据需要扩充这个字典，特别是对于 YOLOV8n 可能检测到的物体
OBJECT_TRANSLATIONS = {
    "person": "人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "bus": "巴士",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "红绿灯",
    "fire hydrant": "消防栓",
    "stop sign": "停车牌",
    "parking meter": "停车计时器",
    "bench": "长椅",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skis": "滑雪板",
    "snowboard": "滑雪板",
    "sports ball": "球",
    "kite": "风筝",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "冲浪板",
    "tennis racket": "网球拍",
    "bottle": "瓶子",
    "wine glass": "高脚杯",
    "cup": "杯子", # <-- 关键修改：将 "cup" 映射为 "杯子"
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "笔记本电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷",
    # ... 更多物体名称可以继续添加 ...
}

# 创建 Flask 应用实例
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_here_please_change_me' # 用于 Flask-SocketIO，请替换为强密钥
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 如果上传目录不存在，则创建它
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 初始化 SocketIO
socketio = SocketIO(app)

# 线程池用于智普清言 API 查询，避免阻塞主线程
API_QUERY_THREAD_POOL = ThreadPoolExecutor(max_workers=5)

# --- 全局应用状态变量 (在多线程环境中需要同步访问) ---
yolo_model = None
zhipu_client = None
cap = None # OpenCV 视频捕获对象
is_running = False # 视频流是否正在运行
processing_frame = False # 标记是否正在处理帧，防止重复处理
video_thread = None # 视频处理线程
current_frame = None # 存储最新捕获的原始帧 (BGR格式)
current_annotated_frame_jpeg = None # 存储最新标注后的 JPEG 图像字节，用于网页显示
detected_objects = set() # 存储当前帧检测到的物体名称
zhipu_info_cache = {} # 缓存智普清言物体信息，避免重复查询
chat_history_messages = [ # 智普清言对话历史
    {"role": "system", "content": ""}
]
video_source_path = 0 # 默认视频源为摄像头 0

# 线程锁，用于保护共享状态变量的并发访问
frame_lock = threading.Lock()
detected_objects_lock = threading.Lock()
zhipu_info_lock = threading.Lock()
chat_history_lock = threading.Lock()

# --- 辅助函数定义 ---

def load_yolo_model(model_path):
    """加载 YOLOv8 模型"""
    global yolo_model
    try:
        yolo_model = YOLO(model_path)
        print(f"YOLO模型 '{model_path}' 加载成功。")
        return yolo_model
    except Exception as e:
        print(f"加载YOLO模型失败: {e}")
        return None

def initialize_zhipu_client(api_key_to_use):
    """初始化智普清言客户端"""
    global zhipu_client

    # 打印接收到的API Key的部分内容，用于调试和确认
    if api_key_to_use:
        print(f"DEBUG: initialize_zhipu_client 收到 API Key (前8/后8字符): {api_key_to_use[:8]}...{api_key_to_use[-8:]}")
        print(f"DEBUG: API Key 完整长度: {len(api_key_to_use)}")
    else:
        print("DEBUG: initialize_zhipu_client 收到 API Key 为空或None。")

    if not api_key_to_use or api_key_to_use == API_KEY_WARNING_PLACEHOLDER:
        print("警告: 智普AI API Key未配置或为默认值，智普API功能将无法使用。")
        zhipu_client = None
        return None

    try:
        print("DEBUG: 尝试实例化 ZhipuAI 客户端...")
        client = ZhipuAI(api_key=api_key_to_use)
        # 这一段测试代码是为了更深入调试，如果不需要可以注释掉，因为它会增加启动时间
        # try:
        #     # 这是一个简单的测试，请求一个空的对话，看是否报错
        #     # 仅用于调试，实际应用中不需要
        #     _ = client.chat.completions.create(
        #         model="glm-4",
        #         messages=[{"role": "user", "content": "hi"}],
        #         temperature=0.7,
        #         top_p=0.9
        #     )
        #     print("DEBUG: 智普清言客户端通过测试调用，确认连接正常。")
        # except Exception as test_e:
        #     print(f"DEBUG: 智普清言客户端测试调用失败: {test_e}")
        #     # 如果测试失败，我们认为初始化是失败的
        #     raise # 重新抛出异常，让外部的except块捕获

        zhipu_client = client
        print("智普清言客户端初始化成功。")
        return client
    except Exception as e:
        # 捕获并打印 ZhipuAI 实例化时抛出的具体异常
        print(f"错误: 初始化智普清言客户端失败。请检查API Key是否正确、网络连接是否畅通，或Key是否过期。详细错误: {e}")
        zhipu_client = None
        return None

def _perform_detection(frame, conf_threshold=0.25, iou_threshold=0.45):
    """执行物体检测"""
    global yolo_model
    if yolo_model is None:
        return frame, set() # 如果模型未加载，返回原始帧和空集合

    results = yolo_model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        max_det=100
    )

    annotated_frame = frame.copy()
    current_detected_objects = set()

    if results and len(results) > 0:
        r = results[0]
        # YOLOv8的plot()方法已经包含了绘制和NMS
        annotated_frame = r.plot()

        # 提取物体类别
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]
                current_detected_objects.add(class_name)

    return annotated_frame, current_detected_objects

def _get_zhipu_object_info(object_name_english):
    """
    使用智普清言 AI 查询物体信息。
    会先将物体名称翻译为中文，并使用更精确的提示词。
    """
    global zhipu_client, zhipu_info_cache
    if zhipu_client is None:
        return "智普清言API客户端未初始化，无法查询信息。"

    # 1. 尝试翻译物体名称为中文
    # 使用 .lower() 确保匹配字典中的小写键
    object_name_chinese = OBJECT_TRANSLATIONS.get(object_name_english.lower(), object_name_english)

    with zhipu_info_lock: # 访问缓存前加锁
        # 缓存键仍然使用英文名称，因为它是从检测结果中获得的
        if object_name_english in zhipu_info_cache:
            return f"(来自缓存) {zhipu_info_cache[object_name_english]}"

    print(f"正在查询 '{object_name_english}' (中文: '{object_name_chinese}') 的相关信息...")

    try:
        # 2. 构建精确的中文提示词和系统角色
        # 系统角色：设定AI的身份和行为规范
        system_prompt = "你是一个严谨的AI助手，专注于提供物体或生物的详细、准确、纯粹的中文信息，不进行闲聊或无关的解释。请避免提及任何英文缩写或英文名称。"

        # 用户提问：明确要求中文描述
        if object_name_chinese == object_name_english:
            # 如果没有找到翻译，仍然使用中文提示，但用英文原词，并引导AI通用描述
            user_prompt = f"请详细描述'{object_name_english}'的生物学特征或作用，并用中文回答。如果这不是常见的中文物体名称，请尝试提供与其功能或形态相关的通用中文描述。不要提及这是一个英文词汇。"
            print(f"注意: 未找到 '{object_name_english}' 的中文翻译，将使用英文原词向智普AI提问并要求中文描述。")
        else:
            user_prompt = f"请详细描述'{object_name_chinese}'的生物学特征或作用，并用中文回答。"
            print(f"将 '{object_name_english}' 翻译为 '{object_name_chinese}' 并向智普AI提问。")

        response = zhipu_client.chat.completions.create(
            model="glm-4", # 或您使用的其他模型，例如 glm-4-0520
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, # 降低温度以获取更事实性的、不发散的回复
            max_tokens=500 # 限制回复长度
        )
        info = response.choices[0].message.content
        with zhipu_info_lock: # 更新缓存前加锁
            zhipu_info_cache[object_name_english] = info # 缓存键依然是英文名称
        print(f"'{object_name_english}' 信息查询成功。")
        return info
    except Exception as e:
        print(f"查询 '{object_name_english}' 信息失败: {e}")
        # 返回时也考虑是否已翻译
        display_name = object_name_chinese if object_name_chinese != object_name_english else object_name_english
        return f"无法获取 '{display_name}' 的信息。错误: {e}"

def _perform_chat_api_call(messages_history_copy):
    """在单独的线程中调用智普清言进行对话，并将结果通过 SocketIO 发送"""
    global zhipu_client, chat_history_messages
    try:
        if zhipu_client is None:
            socketio.emit('chat_response', {'success': False, 'message': "智普清言API客户端未初始化，无法对话。"}, namespace='/')
            return

        # 在传递给智普AI的messages_history_copy中添加或修改系统消息
        # 确保系统消息在列表的开始，并指导AI使用中文回复
        # 首先移除可能存在的旧系统消息（例如，如果messages_history_copy是直接传递的全局副本）
        processed_messages = []
        system_message_exists = False
        for msg in messages_history_copy:
            if msg["role"] == "system":
                # 如果系统消息已经存在，更新其内容并标记
                processed_messages.append({"role": "system", "content": "你是一个能够进行多模态交互的智能AI助手，能够理解用户关于视频内容、物体识别以及日常对话的提问。请用简洁、清晰的**中文**进行回复，并尽力满足用户的要求。"})
                system_message_exists = True
            else:
                processed_messages.append(msg)

        if not system_message_exists:
            # 如果没有系统消息，在最前面添加一个
            processed_messages.insert(0, {"role": "system", "content": "你是一个能够进行多模态交互的智能AI助手，能够理解用户关于视频内容、物体识别以及日常对话的提问。请用简洁、清晰的**中文**进行回复，并尽力满足用户的要求。"})


        response = zhipu_client.chat.completions.create(
            model="glm-4", # 使用文本对话模型
            messages=processed_messages, # 使用处理过的消息列表
            temperature=0.7,
            top_p=0.9
        )
        bot_response_content = response.choices[0].message.content
        with chat_history_lock: # 更新全局对话历史前加锁
            chat_history_messages.append({"role": "assistant", "content": bot_response_content})
        socketio.emit('chat_response', {'success': True, 'message': bot_response_content}, namespace='/')
    except Exception as e:
        error_message = f"智普清言对话失败: {e}"
        print(error_message)
        socketio.emit('chat_response', {'success': False, 'message': error_message}, namespace='/')
    finally:
        socketio.emit('chat_status', {'message': "", 'color': 'black'}, namespace='/') # 清除思考状态

# --- 视频处理线程函数 ---
def _process_video_frames_threaded():
    """在后台线程中处理视频帧，进行检测并更新全局状态"""
    global cap, is_running, processing_frame, current_frame, \
           current_annotated_frame_jpeg, detected_objects, video_source_path

    print("视频处理线程启动。") # 用于调试
    while is_running and cap and cap.isOpened():
        with frame_lock: # 检查是否允许处理新帧
            if processing_frame:
                time.sleep(0.01) # 避免 CPU 空转
                continue
            processing_frame = True # 标记为正在处理

        try:
            ret, frame = cap.read()
            if not ret:
                print(f"视频流结束或无法读取帧。当前源: {video_source_path}") # 更详细的消息
                if not isinstance(video_source_path, int): # 如果是文件，尝试循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 重置到视频开头
                    ret, frame = cap.read() # 尝试再次读取
                    if not ret: # 如果即使循环也失败了
                        print("错误: 循环播放视频失败，或者文件损坏。")
                        is_running = False # 设置标志以停止线程
                        break # 退出循环
                else: # 如果是摄像头，读取失败则停止
                    print("错误: 摄像头无法读取帧，可能是断开或被占用。")
                    is_running = False # 设置标志以停止线程
                    break # 退出循环

            # 如果成功读取到帧 (或循环后成功读取)
            if ret:
                # 存储原始帧的副本，用于单帧分析
                with frame_lock:
                    current_frame = frame.copy()

                # 执行物体检测
                annotated_frame, current_detected_objects_this_frame = _perform_detection(
                    frame.copy(), # 传入帧的副本
                    conf_threshold=0.25,
                    iou_threshold=0.45
                )

                # 将标注后的帧编码为 JPEG 格式，用于网页显示
                ret_jpeg, jpeg = cv2.imencode('.jpg', annotated_frame)
                if not ret_jpeg:
                    print("错误: 无法将帧编码为 JPEG。")
                    continue # 如果编码失败，跳过这一帧

                with frame_lock:
                    current_annotated_frame_jpeg = jpeg.tobytes()

                # 更新检测到的物体列表，并通过 SocketIO 广播到前端
                with detected_objects_lock:
                    if current_detected_objects_this_frame != detected_objects:
                        detected_objects = current_detected_objects_this_frame
                        socketio.emit('detected_objects_update', {'objects': sorted(list(detected_objects))}, namespace='/')
            else: # 如果经过所有尝试后 ret 仍然是 False，确保循环退出
                is_running = False
                break

        except Exception as e:
            print(f"处理视频帧时出错: {e}")
            is_running = False # 发生异常时停止线程
            break # 退出循环
        finally:
            with frame_lock:
                processing_frame = False # 标记处理完成

        time.sleep(0.03) # 控制帧率，大约 30 fps

    print("视频处理线程即将退出。") # 调试信息
    # 当视频线程退出其循环时，它不应该直接调用 reset_application_state() 或 stop_stream_internal()。
    # 它只需允许线程完成其执行。清理工作 (释放 cap 等) 应由主线程调用 stop_stream_internal() 来管理。

    # 然而，如果线程因此方式退出，我们仍然需要确保 cap 被释放。
    if cap: # 确保在线程结束时释放资源
        cap.release()
        print("视频捕获对象已在线程内部释放。")
    # cap = None # 不在这里将 cap 设为 None，由 stop_stream_internal 统一管理
    if not is_running: # 如果是非正常停止，通知前端
        socketio.emit('status_update', {'message': "视频流已停止 (因错误或结束)。"}, namespace='/')


# --- Flask 路由和应用程序逻辑 ---

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html',
                           zhipu_api_key_configured=(ZHIPU_API_KEY != API_KEY_WARNING_PLACEHOLDER))

def generate_frames():
    """生成视频帧的生成器函数，用于 MJPEG 流"""
    global current_annotated_frame_jpeg
    while True:
        with frame_lock:
            if current_annotated_frame_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_annotated_frame_jpeg + b'\r\n')
        time.sleep(0.05) # 控制流传输速率

@app.route('/video_feed')
def video_feed():
    """视频流路由，返回 MJPEG 格式的视频流"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def allowed_file(filename):
    """检查上传文件是否为允许的文件类型"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """处理视频文件上传"""
    if 'video_file' not in request.files:
        return jsonify(status="error", message="没有文件部分")
    file = request.files['video_file']
    if file.filename == '':
        return jsonify(status="error", message="没有选择文件")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify(status="success", message="文件上传成功", filepath=filepath)
    return jsonify(status="error", message="不允许的文件类型")


@app.route('/start_stream', methods=['POST'])
def start_stream():
    """启动或停止视频流"""
    global cap, is_running, video_thread, video_source_path

    if is_running: # 如果已经在运行，则视为停止请求
        stop_stream_internal()
        return jsonify(status="stopped", message="视频流已停止。")

    data = request.json
    source_type = data.get('source_type')
    path = data.get('path') # 文件路径（如果选择文件源）

    if source_type == "webcam":
        video_source_path = 0
        cap = cv2.VideoCapture(0)
    elif source_type == "file":
        if not path or not os.path.exists(path):
            return jsonify(status="error", message="请选择一个有效的视频文件。")
        video_source_path = path
        cap = cv2.VideoCapture(path)
    else:
        return jsonify(status="error", message="无效的视频源类型。")

    if not cap.isOpened():
        return jsonify(status="error", message="无法打开视频源。请检查摄像头是否可用或视频文件路径是否正确。")

    is_running = True
    reset_application_display_data() # 清空旧的显示数据
    socketio.emit('clear_displays', namespace='/') # 通知前端清空显示
    # 重新发送智普AI系统消息，以保持对话连贯性
    with chat_history_lock:
        # 确保初始系统消息是 chat_history_messages 的第一个元素
        initial_message = chat_history_messages[0]["content"]
    socketio.emit('chat_response', {'success': True, 'message': initial_message, 'is_system': True}, namespace='/')


    video_thread = threading.Thread(target=_process_video_frames_threaded, daemon=True)
    video_thread.start()

    socketio.emit('status_update', {'message': "视频流已启动。"}, namespace='/')
    return jsonify(status="started", message="视频流已启动。")

def stop_stream_internal():
    """内部函数：停止视频流"""
    global is_running, video_thread, cap
    is_running = False # 设置停止标志

    # 只有当 video_thread 存在、活跃且不是当前调用线程时才尝试 join
    if video_thread and video_thread.is_alive() and threading.current_thread() != video_thread:
        print("尝试等待视频处理线程优雅退出...") # 调试信息
        video_thread.join(timeout=2.0) # 给线程一些时间来退出
        if video_thread.is_alive():
            print("警告: 视频处理线程未能在规定时间内退出。")
    elif threading.current_thread() == video_thread:
        print("警告: 视频处理线程尝试join自身，已跳过。") # 记录这种错误情况

    if cap:
        cap.release() # 释放视频捕获对象
        cap = None
    # 释放线程引用，这很重要，允许下次重新创建线程对象
    video_thread = None
    socketio.emit('status_update', {'message': "视频流已停止。"}, namespace='/')
    print("视频流已停止。")

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """分析当前帧并查询智普清言"""
    global current_frame, detected_objects, processing_frame

    if current_frame is None:
        return jsonify(status="error", message="当前没有可用于分析的帧。请先启动视频流并等待帧捕获。")

    # 如果视频流正在运行，先停止它
    if is_running:
        stop_stream_internal()
        time.sleep(0.5) # 等待线程完全停止

    with frame_lock: # 确保在处理前加锁
        if processing_frame:
            return jsonify(status="info", message="正在处理帧，请稍后再试。")
        processing_frame = True # 标记正在处理

    try:
        # 对当前帧进行物体检测
        annotated_frame, current_detected_objects_this_frame = _perform_detection(
            current_frame.copy(),
            conf_threshold=0.3, # 分析单帧时可以提高阈值
            iou_threshold=0.5
        )

        # 更新检测到的物体列表，并通过 SocketIO 广播
        with detected_objects_lock:
            detected_objects = current_detected_objects_this_frame
            socketio.emit('detected_objects_update', {'objects': sorted(list(detected_objects))}, namespace='/')

        # 更新网页上显示的帧
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        with frame_lock:
            global current_annotated_frame_jpeg
            current_annotated_frame_jpeg = jpeg.tobytes()

        # 清空智普信息显示区域
        socketio.emit('zhipu_info_update', {'clear': True}, namespace='/')

        # 为检测到的物体查询智普清言信息
        queried_for_this_frame = set()
        if zhipu_client: # 只有当智普客户端初始化成功时才查询
            for obj_name in current_detected_objects_this_frame:
                if obj_name not in queried_for_this_frame:
                    queried_for_this_frame.add(obj_name)
                    # 在后台任务中执行智普查询并发送结果
                    # 这里的 `obj_name` 是 YOLO 检测到的英文名称，会传递给 _get_zhipu_object_info 进行翻译
                    API_QUERY_THREAD_POOL.submit(lambda name: socketio.start_background_task(_query_object_info_and_emit, name), obj_name)
        else:
            socketio.emit('zhipu_info_update', {'info': "智普清言API未初始化，无法查询物体信息。\n\n"}, namespace='/')


        print(f"当前帧分析完成，检测到 {len(current_detected_objects_this_frame)} 种物体: {list(current_detected_objects_this_frame)}")
        socketio.emit('status_update', {'message': f"分析完成，检测到 {len(current_detected_objects_this_frame)} 种物体。"}, namespace='/')
        return jsonify(status="success", message="当前帧分析完成。")

    except Exception as e:
        print(f"分析当前帧失败: {e}")
        socketio.emit('status_update', {'message': f"分析失败: {e}"}, namespace='/')
        return jsonify(status="error", message=f"分析当前帧失败: {e}")
    finally:
        with frame_lock:
            processing_frame = False # 标记处理完成

def _query_object_info_and_emit(obj_name):
    """查询物体信息并发送到前端"""
    # obj_name 此时是英文名称，会传入 _get_zhipu_object_info
    info = _get_zhipu_object_info(obj_name)
    # 在前端显示时，使用原始的英文名称作为标题，信息内容是中文的
    socketio.emit('zhipu_info_update', {'info': f"物体: {obj_name}\n介绍: {info}\n{'-' * 50}\n"}, namespace='/')

@app.route('/reset_app', methods=['POST'])
def reset_app():
    """将应用程序重置到初始状态"""
    reset_application_state() # 调用内部重置函数
    socketio.emit('clear_displays', namespace='/') # 通知前端清空显示
    socketio.emit('status_update', {'message': "应用程序已重置。"}, namespace='/')
    return jsonify(status="success", message="应用程序已重置。")

def reset_application_state():
    """内部函数：重置所有全局应用状态变量"""
    global is_running, processing_frame, cap, video_thread, \
           current_frame, current_annotated_frame_jpeg, detected_objects, \
           zhipu_info_cache, chat_history_messages, video_source_path

    stop_stream_internal() # 确保视频流和线程停止

    with frame_lock:
        current_frame = None
        current_annotated_frame_jpeg = None

    with detected_objects_lock:
        detected_objects = set()

    with zhipu_info_lock:
        zhipu_info_cache.clear()

    with chat_history_lock:
        chat_history_messages = [ # 重置对话历史为初始系统消息
            {"role": "system", "content": ""
                                          "现在可以开始对话啦"}
        ]
    video_source_path = 0 # 重置为默认摄像头

    print("应用程序状态已重置。")

def reset_application_display_data():
    """仅重置显示相关的数据，不停止视频流"""
    global current_frame, current_annotated_frame_jpeg, detected_objects, zhipu_info_cache
    # 注意：chat_history_messages 不在这里重置，它由 reset_application_state 管理

    with frame_lock:
        current_frame = None
        current_annotated_frame_jpeg = None
    with detected_objects_lock:
        detected_objects = set()
    with zhipu_info_lock:
        zhipu_info_cache.clear()


# --- SocketIO 事件处理函数 ---

@socketio.on('connect', namespace='/')
def handle_connect():
    """客户端连接时触发"""
    print('Client connected')
    # 客户端连接时发送初始智普AI系统消息
    with chat_history_lock:
        initial_message = chat_history_messages[0]["content"]
    emit('chat_response', {'success': True, 'message': initial_message, 'is_system': True}, namespace='/')
    emit('status_update', {'message': "服务器已连接。"}, namespace='/')

@socketio.on('disconnect', namespace='/')
def handle_disconnect():
    """客户端断开连接时触发"""
    print('Client disconnected')

@socketio.on('send_chat_message', namespace='/')
def handle_chat_message(data):
    """处理客户端发送的聊天消息"""
    user_message = data['message']
    if not user_message:
        return

    # 将用户消息添加到全局对话历史
    with chat_history_lock:
        chat_history_messages.append({"role": "user", "content": user_message})

    # 立即将用户消息显示在前端
    emit('chat_response', {'success': True, 'user_message': user_message}, namespace='/')
    emit('chat_status', {'message': "AI: 正在思考...", 'color': 'blue'}, namespace='/')

    # 提交智普AI API 调用到线程池，并在后台任务中处理 SocketIO 发送
    with chat_history_lock:
        # 传递对话历史的副本给后台任务，避免并发修改问题
        history_copy = list(chat_history_messages)
    API_QUERY_THREAD_POOL.submit(lambda h: socketio.start_background_task(_perform_chat_api_call, h), history_copy)


# --- 主程序入口 ---
if __name__ == '__main__':
    # 确保在这里打印最终的 ZHIPU_API_KEY 变量的值，以确认没有被意外修改
    if ZHIPU_API_KEY:
        print(f"DEBUG: main函数中最终使用的 ZHIPU_API_KEY (前8/后8字符): {ZHIPU_API_KEY[:8]}...{ZHIPU_API_KEY[-8:]}")
        print(f"DEBUG: main函数中最终使用的 ZHIPU_API_KEY 完整长度: {len(ZHIPU_API_KEY)}")
    else:
        print("DEBUG: main函数中最终使用的 ZHIPU_API_KEY 为空或None。")

    # 应用启动时初始化模型和客户端
    load_yolo_model(YOLO_MODEL_PATH)
    initialize_zhipu_client(ZHIPU_API_KEY)

    # 初始重置，确保应用状态干净
    reset_application_state()

    # 启动 Flask-SocketIO 服务器
    print("Web application starting on http://127.0.0.1:7000")
    # 禁用 Werkzeug 的自动重载器，以避免与 eventlet 冲突，同时保留 debug 模式
    socketio.run(app, host='0.0.0.0', port=7000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

