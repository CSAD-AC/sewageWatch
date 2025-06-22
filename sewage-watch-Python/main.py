import cv2
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import logging

from ultralytics import YOLO

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoStreamer:
    def __init__(self, video_source, model_path="public/yolov8n.pt"):
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
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1 / fps if fps > 0 else 0.033

            while not self.should_stop and websocket.client_state == WebSocketState.CONNECTED:
                start_time = asyncio.get_event_loop().time()

                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # 处理帧 - 直接调用实例方法
                processed_frame = self.process_frame(frame)

                # 编码和发送
                await self.send_frame(websocket, processed_frame, fps)

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

# 运行应用时指定端口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
