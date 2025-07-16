import cv2
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
import math
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import logging
import threading
import queue
import time
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pymysql
import datetime
import uuid
import shutil
from typing import List, Optional

from ultralytics import YOLO
from detection import DetectionProcessor

# 保存原始环境变量值，以便在程序退出时恢复
original_ffmpeg_options = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建临时目录来存储上传的文件
TEMP_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# 读取XML配置文件
def load_config():
    """从XML配置文件中加载配置"""
    try:
        config_path = Path(__file__).parent / "../config.xml"
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}，将使用默认配置")
            return {
                "rtmp_url": "rtmp://example.com/live/default_stream",
                "buffer_size": 30,
                "fps": 60,
                "reconnect_delay": 5,
                "timeout": 10,
                "db_host": "localhost",
                "db_port": 3306,
                "db_user": "root",
                "db_password": "",
                "db_name": "sewagewatch",
                "history_path": "../history",
                "detect_types": ["bottle", "bird"]
            }
        
        tree = ET.parse(config_path)
        root = tree.getroot()
        
        # 读取RTMP URL
        rtmp_url = root.find("rtmp/url").text
        
        # 读取其他设置
        settings = root.find("settings")
        buffer_size = int(settings.find("buffer_size").text)
        fps = int(settings.find("fps").text)
        reconnect_delay = int(settings.find("reconnect_delay").text)
        timeout = int(settings.find("timeout").text)
        
        # 读取数据库配置
        database = root.find("database")
        db_host = database.find("host").text
        db_port = int(database.find("port").text)
        db_user = database.find("user").text
        db_password = database.find("password").text
        db_name = database.find("name").text
        
        # 读取历史记录配置
        history = root.find("history")
        history_path = history.find("storage_path").text
        detect_types = [type_elem.text for type_elem in history.find("detect_types").findall("type")]
        
        logger.info(f"已从配置文件加载RTMP URL: {rtmp_url}")
        logger.info(f"已从配置文件加载数据库配置: {db_host}:{db_port}")
        logger.info(f"已从配置文件加载历史记录配置: {history_path}, 检测类型: {detect_types}")
        
        return {
            "rtmp_url": rtmp_url,
            "buffer_size": buffer_size,
            "fps": fps,
            "reconnect_delay": reconnect_delay,
            "timeout": timeout,
            "db_host": db_host,
            "db_port": db_port,
            "db_user": db_user,
            "db_password": db_password,
            "db_name": db_name,
            "history_path": history_path,
            "detect_types": detect_types
        }
    except Exception as e:
        logger.error(f"读取配置文件时出错: {e}")
        return {
            "rtmp_url": "rtmp://example.com/live/default_stream",
            "buffer_size": 30,
            "fps": 60,
            "reconnect_delay": 5,
            "timeout": 10,
            "db_host": "localhost",
            "db_port": 3306,
            "db_user": "root",
            "db_password": "",
            "db_name": "sewagewatch",
            "history_path": "../history",
            "detect_types": ["bottle", "bird"]
        }

# 加载配置
config = load_config()

# 确保历史记录目录存在
os.makedirs(config["history_path"], exist_ok=True)

# 数据库连接函数
def get_db_connection():
    """创建并返回数据库连接"""
    try:
        connection = pymysql.connect(
            host=config["db_host"],
            port=config["db_port"],
            user=config["db_user"],
            password=config["db_password"],
            database=config["db_name"],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return None

# 初始化数据库表
def init_database():
    """确保数据库表存在"""
    try:
        connection = get_db_connection()
        if connection:
            with connection.cursor() as cursor:
                # 创建历史记录表
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '历史ID',
                    taskId INT COMMENT '关联的任务ID',
                    type VARCHAR(20) NOT NULL COMMENT '污染类型',
                    src VARCHAR(255) NOT NULL COMMENT '污染图片路径',
                    createdTime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史记录表';
                """)
                connection.commit()
                logger.info("数据库表初始化成功")
            connection.close()
        else:
            logger.error("无法初始化数据库表，连接失败")
    except Exception as e:
        logger.error(f"初始化数据库表失败: {e}")

# 保存历史记录到数据库
def save_history_record(type_name, image_path, task_id=None):
    """保存历史记录到数据库"""
    try:
        connection = get_db_connection()
        if connection:
            with connection.cursor() as cursor:
                # 插入历史记录
                sql = """
                INSERT INTO history (taskId, type, src, createdTime)
                VALUES (%s, %s, %s, %s)
                """
                now = datetime.datetime.now()
                cursor.execute(sql, (task_id, type_name, image_path, now))
                connection.commit()
                logger.info(f"历史记录已保存: {type_name}, {image_path}")
            connection.close()
            return True
        else:
            logger.error("无法保存历史记录，数据库连接失败")
            return False
    except Exception as e:
        logger.error(f"保存历史记录失败: {e}")
        return False

def apply_h264_optimizations():
    """设置环境变量以优化FFmpeg的H.264解码"""
    ffmpeg_options = {
        'rtsp_transport': 'tcp',          # 强制使用TCP，保证数据传输可靠性
        'probesize': '10000000',        # 增加探测数据大小 (10MB)，帮助FFmpeg更好地识别流信息
        'analyzeduration': '10000000',  # 增加分析时长 (10秒)，在建立连接时分析更多数据
        'flags': 'low_delay',             # 开启低延迟标志，减少缓冲
        'fflags': 'nobuffer+fastseek+flush_packets',  # 禁用缓冲，快速寻址，立即刷新数据包
        'max_delay': '0',                 # 最大延迟设为0
        'thread_type': 'frame',           # 使用帧级线程
        'threads': '1',                   # 限制线程数避免竞争
        'err_detect': 'ignore_err',       # 忽略错误继续解码
        'skip_frame': 'nokey'             # 跳过非关键帧错误
    }
    # 注意：在Windows上，分隔符是';'，而在Linux上是':'。这里使用';'。
    env_options = ';'.join([f'{k};{v}' for k, v in ffmpeg_options.items()])
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = env_options
    logger.info(f"已应用增强FFmpeg优化环境变量: {env_options}")

def clear_h264_optimizations():
    """清理环境变量，恢复到原始状态"""
    if original_ffmpeg_options is None:
        if 'OPENCV_FFMPEG_CAPTURE_OPTIONS' in os.environ:
            del os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']
            logger.info("已清理FFmpeg优化环境变量。")
    else:
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = original_ffmpeg_options
        logger.info("已恢复原始FFmpeg环境变量。")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"],
)

# 添加请求日志中间件
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    logger.info(f"开始处理请求: {method} {path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"请求处理完成: {method} {path} - 状态码: {response.status_code} - 耗时: {process_time:.4f}秒")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"请求处理异常: {method} {path} - 耗时: {process_time:.4f}秒 - 错误: {str(e)}")
        raise

# 配置静态文件服务，使前端能够访问历史图片
history_dir = Path(config["history_path"])
app.mount("/history", StaticFiles(directory=str(history_dir)), name="history")

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    apply_h264_optimizations()
    init_database()

@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时执行"""
    clear_h264_optimizations()
    
    # 清理临时上传目录
    try:
        for file in os.listdir(TEMP_UPLOAD_DIR):
            file_path = os.path.join(TEMP_UPLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        logger.info("已清理临时上传目录")
    except Exception as e:
        logger.error(f"清理临时上传目录时出错: {e}")

# 创建检测处理器实例
detection_processor = DetectionProcessor(
    model_path="public/yolov8n_7_11.pt",
    history_path=config["history_path"],
    detect_types=config["detect_types"]
)

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    处理上传的图片文件
    """
    file_path = None
    try:
        logger.info(f"接收到图片上传请求: {file.filename}, 类型: {file.content_type}")
        
        # 检查文件类型
        if not file.content_type.startswith("image/"):
            logger.warning(f"文件类型不支持: {file.content_type}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"只支持图片文件，收到的是: {file.content_type}"}
            )
        
        # 确保临时目录存在
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
        # 生成唯一文件名，避免文件名冲突
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)
        
        # 保存上传的文件到临时目录
        logger.info(f"正在保存上传的图片到: {file_path}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"已保存上传的图片: {file_path}, 大小: {os.path.getsize(file_path)} 字节")
        
        # 处理图片
        logger.info(f"开始处理图片: {file_path}")
        result = detection_processor.process_image(file_path)
        logger.info(f"图片处理完成: {file_path}, 结果: {result.get('success', False)}")
        
        # 如果检测成功，保存到数据库
        if result.get("success", False):
            logger.info("检测成功，正在保存到数据库")
            detection_processor.save_to_database(get_db_connection, result)
            logger.info("已保存到数据库")
        else:
            logger.warning(f"检测未成功: {result.get('error', '未知错误')}")
        
        return result
    except Exception as e:
        logger.error(f"处理上传图片时出错: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"处理图片时出错: {str(e)}"}
        )
    finally:
        # 确保临时文件被删除
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"已删除临时文件: {file_path}")
            except Exception as e:
                logger.error(f"删除临时文件失败: {file_path}, 错误: {e}")

@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    frame_interval: int = Form(30)  # 默认每30帧处理一次
):
    """
    处理上传的视频文件
    """
    file_path = None
    try:
        logger.info(f"接收到视频上传请求: {file.filename}, 类型: {file.content_type}, 帧间隔: {frame_interval}")
        
        # 检查文件类型
        if not file.content_type.startswith("video/"):
            logger.warning(f"文件类型不支持: {file.content_type}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"只支持视频文件，收到的是: {file.content_type}"}
            )
        
        # 确保临时目录存在
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
        # 生成唯一文件名，避免文件名冲突
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)
        
        # 保存上传的文件到临时目录
        logger.info(f"正在保存上传的视频到: {file_path}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"已保存上传的视频: {file_path}, 大小: {os.path.getsize(file_path)} 字节")
        
        # 处理视频
        logger.info(f"开始处理视频: {file_path}, 帧间隔: {frame_interval}")
        result = detection_processor.process_video(file_path, frame_interval=frame_interval)
        logger.info(f"视频处理完成: {file_path}, 结果: {result.get('success', False)}")
        
        # 如果检测成功，保存到数据库
        if result.get("success", False):
            logger.info("检测成功，正在保存到数据库")
            detection_processor.save_to_database(get_db_connection, result)
            logger.info("已保存到数据库")
        else:
            logger.warning(f"检测未成功: {result.get('error', '未知错误')}")
        
        return result
    except Exception as e:
        logger.error(f"处理上传视频时出错: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"处理视频时出错: {str(e)}"}
        )
    finally:
        # 确保临时文件被删除
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"已删除临时文件: {file_path}")
            except Exception as e:
                logger.error(f"删除临时文件失败: {file_path}, 错误: {e}")

class VideoStreamer:
    def __init__(self, video_source, model_path="public/yolov8n_7_11.pt"):
        self.video_source = video_source
        self.cap = None
        self.should_stop = False
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        """使用YOLOv8处理视频帧并绘制检测结果"""
        # 模型推理
        results = self.model(frame, conf=0.4, iou=0.5)  # 设置置信度和IOU阈值

        # 获取检测结果
        result = results[0]  # 单帧结果

        # 修改检测结果中的bird标签为bottle
        for box in result.boxes:
            cls = int(box.cls)
            if result.names[cls] == 'bird':
                result.names[cls] = 'bottle'
        
        # 检查是否需要记录所有类型（配置中包含*）
        record_all_types = '*' in config["detect_types"]
        
        # 收集所有检测到的物体类型
        all_detected_types = {}
        for box in result.boxes:
            cls = int(box.cls)
            type_name = result.names[cls]
            if type_name not in all_detected_types:
                all_detected_types[type_name] = 1
            else:
                all_detected_types[type_name] += 1
        
        # 确定需要记录的类型
        detected_types_to_record = {}
        if record_all_types and all_detected_types:
            # 如果配置中有*且检测到物体，记录所有类型
            detected_types_to_record = all_detected_types
        else:
            # 否则只记录配置中指定的类型
            for box in result.boxes:
                cls = int(box.cls)
                type_name = result.names[cls]
                if type_name in config["detect_types"]:
                    if type_name not in detected_types_to_record:
                        detected_types_to_record[type_name] = 1
                    else:
                        detected_types_to_record[type_name] += 1
        
        # 在原图上绘制边界框和标签
        annotated_frame = result.plot(
            conf=True,  # 显示置信度
            line_width=2,  # 边界框线条宽度
            font_size=12  # 标签字体大小
        )
        
        # 如果检测到需要记录的类型，保存图片并记录到数据库
        if detected_types_to_record:
            # 保存图片
            now = datetime.datetime.now()
            date_str = now.strftime("%Y%m%d_%H%M%S")
            
            # 生成唯一文件名
            unique_id = str(uuid.uuid4()).replace("-", "")
            filename = f"{date_str}_{unique_id}.jpg"
            image_path = os.path.join(config["history_path"], filename)
            
            # 保存图片
            cv2.imwrite(image_path, annotated_frame)
            
            # 将所有检测到的类型合并为一个字符串，用逗号分隔
            all_types_str = ",".join(all_detected_types.keys())
            
            # 记录到数据库
            relative_path = f"/history/{filename}"  # 使用相对路径存储
            save_history_record(all_types_str, relative_path)
            
            logger.info(f"已记录历史: 类型={all_types_str}, 图片={image_path}")

        # 获取检测统计信息
        detections = len(result.boxes)

        # 添加统计信息到画面
        cv2.putText(
            annotated_frame,
            f"Detections: {detections}",
            (10, 30),  # 位置
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            1,  # 字体大小
            (0, 255, 0),  # 颜色（绿）
            2,  # 线条宽度
            cv2.LINE_AA  # 抗锯齿
        )

        return annotated_frame

    async def initialize(self):
        """初始化视频捕获"""
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开视频源: {self.video_source}")
            return True
        except Exception as e:
            logger.error(f"初始化视频捕获失败: {e}")
            return False

    async def stream_video(self, websocket):
        """流式传输视频帧"""
        try:
            # 锁定帧率为30fps
            fps = 30.0
            frame_delay = 0.033  # 固定30fps

            while not self.should_stop and websocket.client_state == WebSocketState.CONNECTED:
                start_time = asyncio.get_event_loop().time()

                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # 处理帧 - 直接调用实例方法
                processed_frame = self.process_frame(frame)

                # 编码和发送
                await self.send_frame(websocket, processed_frame, 30.0)

                # 控制帧率
                elapsed = asyncio.get_event_loop().time() - start_time
                await asyncio.sleep(max(0, frame_delay - elapsed))

        except Exception as e:
            logger.error(f"视频流错误: {e}")
        finally:
            self.release()

    async def send_frame(self, websocket, frame, fps):
        """发送帧到客户端"""
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        await websocket.send_json({
            "image": frame_b64,
            "fps": round(fps, 1),
            "speed": round(np.random.uniform(10, 15), 1),
            "weather": "晴朗"
        })

    def release(self):
        """释放资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

class RTMPStreamer:
    def __init__(self, rtmp_url=None, model_path="public/yolov8n.pt"):
        # 如果未提供RTMP URL，则使用配置文件中的URL
        self.rtmp_url = rtmp_url if rtmp_url else config["rtmp_url"]
        self.cap = None
        self.should_stop = False
        self.frame_queue = queue.Queue(maxsize=config["buffer_size"])  # 使用配置的队列大小
        self.capture_thread = None
        self.is_capturing = False
        self.reconnect_delay = config["reconnect_delay"]  # 重连延迟时间
        self.timeout = config["timeout"]  # 超时时间
        
        # 初始化YOLO模型
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO模型初始化成功: {model_path}")
        except Exception as e:
            logger.error(f"YOLO模型初始化失败: {e}")
            self.model = None

    def initialize(self):
        """
        初始化RTMP视频捕获。
        使用在应用启动时设置的环境变量进行优化。
        """
        try:
            logger.info(f"正在使用优化配置初始化RTMP流: {self.rtmp_url}")
            self.cap = cv2.VideoCapture(self.rtmp_url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                raise ValueError(f"无法打开RTMP视频源: {self.rtmp_url}")
            
            # 设置额外的VideoCapture属性来处理H.264解码问题
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区大小
            self.cap.set(cv2.CAP_PROP_FPS, 30)        # 设置期望帧率为30fps
            
            # 尝试读取第一帧以验证连接
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                logger.info(f"RTMP视频源初始化成功，首帧尺寸: {test_frame.shape}")
                return True
            else:
                logger.warning("首帧读取失败，但继续尝试...")
                return True  # 有些流需要几次尝试才能稳定
                
        except Exception as e:
            logger.error(f"初始化RTMP捕获失败: {e}")
            self.release()
            return False
            
    def process_frame(self, frame):
        """使用YOLOv8处理视频帧并绘制检测结果"""
        if self.model is None:
            # 如果模型初始化失败，只添加基本的RTMP标识
            cv2.putText(
                frame,
                "RTMP LIVE (Optimized)",
                (10, 30),  # 位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                1,  # 字体大小
                (0, 0, 255),  # 颜色（红）
                2,  # 线条宽度
                cv2.LINE_AA  # 抗锯齿
            )
            return frame
        
        try:
            # 模型推理
            results = self.model(frame, conf=0.4, iou=0.5)  # 设置置信度和IOU阈值

            # 获取检测结果
            result = results[0]  # 单帧结果

            # 修改检测结果中的bird标签为bottle
            for box in result.boxes:
                cls = int(box.cls)
                # result.names[cls] = 'bottle'
                if result.names[cls] == 'bird':
                    result.names[cls] = 'bottle'
                

            # 检查是否需要记录所有类型（配置中包含*）
            record_all_types = '*' in config["detect_types"]
            
            # 收集所有检测到的物体类型
            all_detected_types = {}
            for box in result.boxes:
                cls = int(box.cls)
                type_name = result.names[cls]
                if type_name not in all_detected_types:
                    all_detected_types[type_name] = 1
                else:
                    all_detected_types[type_name] += 1
            
            # 确定需要记录的类型
            detected_types_to_record = {}
            if record_all_types and all_detected_types:
                # 如果配置中有*且检测到物体，记录所有类型
                detected_types_to_record = all_detected_types
            else:
                # 否则只记录配置中指定的类型
                for box in result.boxes:
                    cls = int(box.cls)
                    type_name = result.names[cls]
                    if type_name in config["detect_types"]:
                        if type_name not in detected_types_to_record:
                            detected_types_to_record[type_name] = 1
                        else:
                            detected_types_to_record[type_name] += 1
            
            # 在原图上绘制边界框和标签
            annotated_frame = result.plot(
                conf=True,  # 显示置信度
                line_width=2,  # 边界框线条宽度
                font_size=12  # 标签字体大小
            )
            
            # 如果检测到需要记录的类型，保存图片并记录到数据库
            if detected_types_to_record:
                # 保存图片
                now = datetime.datetime.now()
                date_str = now.strftime("%Y%m%d_%H%M%S")
                
                # 生成唯一文件名
                unique_id = str(uuid.uuid4()).replace("-", "")
                filename = f"{date_str}_{unique_id}.jpg"
                image_path = os.path.join(config["history_path"], filename)
                
                # 保存图片
                cv2.imwrite(image_path, annotated_frame)
                
                # 将所有检测到的类型合并为一个字符串，用逗号分隔
                all_types_str = ",".join(all_detected_types.keys())
                
                # 记录到数据库
                relative_path = f"..\\history\\{filename}"  # 使用相对路径存储
                save_history_record(all_types_str, relative_path)
                
                logger.info(f"已记录历史: 类型={all_types_str}, 图片={image_path}")

            # 获取检测统计信息
            detections = len(result.boxes)

            # 添加统计信息到画面
            cv2.putText(
                annotated_frame,
                f"Detections: {detections} | RTMP LIVE",
                (10, 30),  # 位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                1,  # 字体大小
                (0, 0, 255),  # 颜色（红）
                2,  # 线条宽度
                cv2.LINE_AA  # 抗锯齿
            )

            return annotated_frame
        except Exception as e:
            logger.error(f"YOLO处理帧时出错: {e}")
            # 出错时返回原始帧，并添加错误标识
            cv2.putText(
                frame,
                "RTMP LIVE (Model Error)",
                (10, 30),  # 位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                1,  # 字体大小
                (0, 0, 255),  # 颜色（红）
                2,  # 线条宽度
                cv2.LINE_AA  # 抗锯齿
            )
            return frame

    def capture_frames(self):
        """在后台线程中捕获RTMP帧，并处理连接中断"""
        if not self.initialize():
            self.is_capturing = False
            logger.error("无法启动RTMP捕获线程，初始化失败。")
            return

        last_successful_frame_time = time.time()
        
        while self.is_capturing and not self.should_stop:
            if not self.cap or not self.cap.isOpened():
                logger.error(f"RTMP连接丢失，将在{self.reconnect_delay}秒后尝试重新连接...")
                time.sleep(self.reconnect_delay)
                if not self.initialize():
                    continue  # 如果重连失败，则在下一次循环继续尝试
                else:
                    last_successful_frame_time = time.time() # 重置计时器

            ret, frame = self.cap.read()

            if ret and frame is not None and frame.size > 0:
                # 验证帧的有效性
                if len(frame.shape) == 3 and frame.shape[0] > 0 and frame.shape[1] > 0:
                    last_successful_frame_time = time.time()
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    else:
                        # 队列已满，丢弃旧帧以减少延迟
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.frame_queue.put(frame)
                else:
                    logger.debug("收到无效帧，跳过...")
            else:
                # 处理解码错误或无帧情况
                current_time = time.time()
                if current_time - last_successful_frame_time > self.timeout:
                    logger.warning(f"超过{self.timeout}秒未收到有效帧，将重新初始化连接。")
                    self.release()
                    # 循环将自动处理重新初始化
                    last_successful_frame_time = current_time # 重置计时器以避免快速连续重连
                else:
                    # 短暂等待，避免CPU占用过高
                    time.sleep(0.01)
        
        self.release()
        logger.info("RTMP捕获线程已停止。")
        
    def start_capture(self):
        """启动RTMP捕获线程"""
        if self.initialize():
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.capture_thread.start()
            logger.info("RTMP捕获线程已启动")
            return True
        return False

    async def stream_video(self, websocket):
        """流式传输RTMP视频帧"""
        try:
            frame_delay = 0.033  # 锁定30fps
            
            while not self.should_stop and websocket.client_state == WebSocketState.CONNECTED:
                start_time = asyncio.get_event_loop().time()
                
                try:
                    # 从队列获取最新帧
                    frame = self.frame_queue.get_nowait()
                    
                    # 处理帧
                    processed_frame = self.process_frame(frame)
                    
                    # 编码和发送
                    await self.send_frame(websocket, processed_frame, 30.0)
                    
                except queue.Empty:
                    # 没有新帧，等待一下
                    await asyncio.sleep(0.01)
                    continue
                
                # 控制帧率
                elapsed = asyncio.get_event_loop().time() - start_time
                await asyncio.sleep(max(0, frame_delay - elapsed))
                
        except Exception as e:
            logger.error(f"RTMP视频流错误: {e}")
        finally:
            self.release()

    async def send_frame(self, websocket, frame, fps):
        """发送帧到客户端"""
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        await websocket.send_json({
            "image": frame_b64,
            "fps": round(fps, 1),
            "speed": round(np.random.uniform(10, 15), 1),
            "weather": "晴朗",
            "source": "RTMP"
        })

    def release(self):
        """安全地释放所有资源"""
        logger.info("开始释放RTMPStreamer资源...")
        self.should_stop = True
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            logger.info("等待捕获线程结束...")
            self.capture_thread.join(timeout=2)
            if self.capture_thread.is_alive():
                logger.warning("捕获线程在超时后仍未结束")
        
        if self.cap:
            if self.cap.isOpened():
                logger.info("释放VideoCapture对象...")
                self.cap.release()
            self.cap = None
        
        # 清空队列
        logger.info("清空帧队列...")
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("RTMPStreamer资源释放完成")

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("进行连接尝试")
    logger.info(f"websocket {websocket}")
    await websocket.accept()
    streamer = VideoStreamer("public/sample.mp4")  # 或使用0表示摄像头

    if not await streamer.initialize():
        await websocket.close(code=1008, reason="无法初始化视频源")
        return

    try:
        await streamer.stream_video(websocket)
    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        streamer.should_stop = True
        streamer.release()
        await websocket.close()

@app.websocket("/ws/rtmp")
async def rtmp_websocket_endpoint(websocket: WebSocket, rtmp_url: str = None):
    """RTMP视频流WebSocket端点，优化了关闭逻辑"""
    if not rtmp_url:
        logger.info(f"未提供RTMP URL参数，将使用配置文件中的默认URL: {config['rtmp_url']}")
    else:
        logger.info(f"RTMP连接尝试，URL: {rtmp_url}")
    
    await websocket.accept()
    
    rtmp_streamer = None
    
    try:
        if rtmp_url:
            rtmp_streamer = RTMPStreamer(rtmp_url)
        else:
            rtmp_streamer = RTMPStreamer()
        
        if not rtmp_streamer.start_capture():
            logger.error("启动RTMP捕获失败")
            # 初始化失败时，确保websocket被关闭
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1008, reason="无法连接RTMP流")
            return
        
        logger.info("RTMP捕获启动成功，开始推流...")
        await rtmp_streamer.stream_video(websocket)
        
    except WebSocketDisconnect:
        logger.info("RTMP客户端断开连接")
    except Exception as e:
        logger.error(f"RTMP WebSocket主循环出现异常: {e}", exc_info=True)
    finally:
        logger.info("进入RTMP WebSocket的finally块")
        if rtmp_streamer:
            rtmp_streamer.release()
        
        # 使用WebSocketState检查连接状态，这是最可靠的方式
        if websocket.client_state == WebSocketState.CONNECTED:
            logger.info("WebSocket连接仍处于打开状态，现在关闭它")
            try:
                await websocket.close(code=1000)
            except RuntimeError as e:
                # 捕获在极少数竞态条件下可能发生的重复关闭错误
                logger.warning(f"尝试关闭WebSocket时发生运行时错误: {e}")
        else:
            logger.info(f"WebSocket连接已处于 {websocket.client_state} 状态，无需关闭")



if __name__ == "__main__":
    # 确保历史记录目录存在
    os.makedirs(config["history_path"], exist_ok=True)
    
    # 初始化数据库
    init_database()
    
    # 启动应用
    import uvicorn

    uvicorn.run(
        app,
        host=config.get("host", "localhost"),
        port=config.get("port", 8081)
    )