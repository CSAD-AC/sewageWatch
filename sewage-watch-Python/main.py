import cv2
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import logging
import threading
import queue
import time
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from ultralytics import YOLO

# 保存原始环境变量值，以便在程序退出时恢复
original_ffmpeg_options = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')

import xml.etree.ElementTree as ET

# 解析XML配置文件
tree = ET.parse('../config.xml')
root = tree.getroot()

# 构建配置字典
config = {
    'rtmp_url': root.find('.//rtmp/url').text,
    'buffer_size': int(root.find('.//settings/buffer_size').text),
    'fps': int(root.find('.//settings/fps').text),
    'reconnect_delay': int(root.find('.//settings/reconnect_delay').text),
    'timeout': int(root.find('.//settings/timeout').text)
}

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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                "timeout": 10
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
        
        logger.info(f"已从配置文件加载RTMP URL: {rtmp_url}")
        
        return {
            "rtmp_url": rtmp_url,
            "buffer_size": buffer_size,
            "fps": fps,
            "reconnect_delay": reconnect_delay,
            "timeout": timeout
        }
    except Exception as e:
        logger.error(f"读取配置文件时出错: {e}")
        return {
            "rtmp_url": "rtmp://example.com/live/default_stream",
            "buffer_size": 30,
            "fps": 60,
            "reconnect_delay": 5,
            "timeout": 10
        }

# 加载配置
config = load_config()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    apply_h264_optimizations()

@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时执行"""
    clear_h264_optimizations()

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

        #######修改
        # 修改检测结果中的bird标签为bottle
        for box in result.boxes:
            cls = int(box.cls)
            if result.names[cls] == 'bird':
                result.names[cls] = 'bottle'
        

        # 在原图上绘制边界框和标签
        annotated_frame = result.plot(
            conf=True,  # 显示置信度
            line_width=2,  # 边界框线条宽度
            font_size=12  # 标签字体大小
        )

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
                if result.names[cls] == 'bird':
                    result.names[cls] = 'bottle'

            # 在原图上绘制边界框和标签
            annotated_frame = result.plot(
                conf=True,  # 显示置信度
                line_width=2,  # 边界框线条宽度
                font_size=12  # 标签字体大小
            )

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

@app.get("/")
async def get_index():
    """返回API信息"""
    return {
        "message": "污水监控视频流API",
        "endpoints": {
            "local_video": "/ws/video",
            "rtmp_stream": "/ws/rtmp?rtmp_url=<RTMP_URL> 或 /ws/rtmp (使用配置文件中的URL)"
        },
        "examples": {
            "custom_rtmp": "/ws/rtmp?rtmp_url=rtmp://example.com/live/stream",
            "default_rtmp": "/ws/rtmp (使用配置的URL: " + config["rtmp_url"] + ")"
        },
        "config_file": "../config.xml"
    }

# 运行应用时指定端口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)