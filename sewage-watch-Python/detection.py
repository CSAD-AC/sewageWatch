import cv2
import os
import uuid
import datetime
import logging
import numpy as np
import threading
import time
from ultralytics import YOLO
from pathlib import Path
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionProcessor:
    """
    封装对图片和视频的模型处理逻辑，支持详细的帧级分析结果
    """
    def __init__(self, model_path="public/yolov8n_7_11.pt", history_path="../history", detect_types=None):
        """
        初始化检测处理器
        
        Args:
            model_path: YOLO模型路径
            history_path: 历史记录保存路径
            detect_types: 需要检测的物体类型列表，如果为None或包含'*'则检测所有类型
        """
        self.history_path = history_path
        self.detect_types = detect_types or ["bottle", "plastic", "trash", "bird"]
        
        # 确保历史记录目录存在
        os.makedirs(self.history_path, exist_ok=True)
        
        # 初始化YOLO模型
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO模型初始化成功: {model_path}")
        except Exception as e:
            logger.error(f"YOLO模型初始化失败: {e}")
            self.model = None

class RTMPRecorder:
    """
    RTMP流录制器，负责将RTMP流录制为MP4文件
    """
    def __init__(self, rtmp_url, output_dir="temp_uploads", max_duration=3600):
        """
        初始化RTMP录制器
        
        Args:
            rtmp_url: RTMP流地址
            output_dir: 输出文件保存目录
            max_duration: 最大录制时长(秒)
        """
        self.rtmp_url = rtmp_url
        self.output_dir = output_dir
        self.max_duration = max_duration
        self.is_recording = False
        self.cap = None
        self.writer = None
        self.output_file = None
        self.start_time = None
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 配置日志
        self.logger = logging.getLogger("RTMPRecorder")
    
    def start_recording(self):
        """
        开始录制RTMP流
        
        Returns:
            str: 录制文件的路径
        """
        try:
            if self.is_recording:
                self.logger.warning("录制已经在进行中")
                return self.output_file
            
            # 初始化视频捕获
            self.cap = cv2.VideoCapture(self.rtmp_url, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开RTMP流: {self.rtmp_url}")
            
            # 获取视频属性
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 生成输出文件名
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4()).replace("-", "")[:8]
            self.output_file = os.path.join(self.output_dir, f"rtmp_recording_{timestamp}_{unique_id}.mp4")
            
            # 定义编码器和创建VideoWriter对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_file, fourcc, fps, (width, height))
            
            # 开始录制
            self.is_recording = True
            self.start_time = time.time()
            self.logger.info(f"开始录制RTMP流到文件: {self.output_file}")
            
            # 开始录制线程
            self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.recording_thread.start()
            
            return self.output_file
        except Exception as e:
            self.logger.error(f"启动录制失败: {str(e)}")
            self.release()
            return None
    
    def _record_loop(self):
        """
        录制循环，在单独线程中运行
        """
        try:
            while self.is_recording and self.cap.isOpened():
                # 检查是否超过最大录制时长
                if time.time() - self.start_time > self.max_duration:
                    self.logger.info(f"达到最大录制时长({self.max_duration}秒)，停止录制")
                    self.stop_recording()
                    break
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.logger.warning("无法读取帧，可能是连接问题")
                    # 短暂暂停避免CPU占用过高
                    time.sleep(0.01)
                    continue
                
                # 写入帧到文件
                self.writer.write(frame)
                
        except Exception as e:
            self.logger.error(f"录制过程中出错: {str(e)}")
        finally:
            self.release()
    
    def stop_recording(self):
        """
        停止录制RTMP流
        
        Returns:
            str: 录制文件的路径
        """
        self.is_recording = False
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2)
        self.release()
        self.logger.info(f"录制已停止，文件保存为: {self.output_file}")
        return self.output_file
    
    def release(self):
        """
        释放资源
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def process_image(self, image_path, save_result=True):
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            save_result: 是否保存检测结果
            
        Returns:
            dict: 包含检测结果的字典
        """
        try:
            if self.model is None:
                return {"success": False, "error": "模型未初始化"}
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": f"无法读取图片: {image_path}"}
            
            # 模型推理
            results = self.model(image, conf=0.4, iou=0.5)
            result = results[0]  # 单帧结果
            
            # 修改检测结果中的bird标签为bottle
            for box in result.boxes:
                cls = int(box.cls)
                if result.names[cls] == 'bird':
                    result.names[cls] = 'bottle'
            
            # 检查是否需要记录所有类型
            record_all_types = '*' in self.detect_types
            
            # 收集所有检测到的物体类型
            all_detected_types = {}
            for box in result.boxes:
                cls = int(box.cls)
                type_name = result.names[cls]
                conf = float(box.conf)
                
                if type_name not in all_detected_types:
                    all_detected_types[type_name] = {
                        "count": 1,
                        "confidence": [conf]
                    }
                else:
                    all_detected_types[type_name]["count"] += 1
                    all_detected_types[type_name]["confidence"].append(conf)
            
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
                    conf = float(box.conf)
                    
                    if type_name in self.detect_types:
                        if type_name not in detected_types_to_record:
                            detected_types_to_record[type_name] = {
                                "count": 1,
                                "confidence": [conf]
                            }
                        else:
                            detected_types_to_record[type_name]["count"] += 1
                            detected_types_to_record[type_name]["confidence"].append(conf)
            
            # 在原图上绘制边界框和标签
            annotated_image = result.plot(
                conf=True,  # 显示置信度
                line_width=2,  # 边界框线条宽度
                font_size=12  # 标签字体大小
            )
            
            # 保存结果图片
            result_path = None
            if save_result and detected_types_to_record:
                # 生成唯一文件名
                now = datetime.datetime.now()
                date_str = now.strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4()).replace("-", "")
                filename = f"{date_str}_{unique_id}.jpg"
                result_path = os.path.join(self.history_path, filename)
                
                # 保存图片
                cv2.imwrite(result_path, annotated_image)
                logger.info(f"已保存检测结果图片: {result_path}")
            
            # 计算每种类型的平均置信度
            for type_name, data in all_detected_types.items():
                data["avg_confidence"] = sum(data["confidence"]) / len(data["confidence"])
                data["confidence"] = [round(c, 2) for c in data["confidence"]]
            
            # 准备返回结果
            detection_result = {
                "success": True,
                "detected_objects": all_detected_types,
                "total_detections": len(result.boxes),
                "result_path": result_path,
                "relative_path": f"/history/{Path(result_path).name}" if result_path else None
            }
            
            return detection_result
            
        except Exception as e:
            logger.error(f"处理图片时出错: {e}")
            return {"success": False, "error": str(e)}
    
    def process_video(self, video_path, save_frames=True, frame_interval=30):
        """
        处理视频文件
        
        Args:
            video_path: 视频路径
            save_frames: 是否保存检测到物体的帧
            frame_interval: 处理帧的间隔（每隔多少帧处理一次）
            
        Returns:
            dict: 包含检测结果的字典
        """
        try:
            if self.model is None:
                return {"success": False, "error": "模型未初始化"}
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": f"无法打开视频: {video_path}"}
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # 初始化结果
            all_detected_types = {}
            saved_frames = []
            frame_index = 0
            
            # 处理视频帧
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每隔frame_interval帧处理一次
                if frame_index % frame_interval == 0:
                    # 模型推理
                    results = self.model(frame, conf=0.4, iou=0.5)
                    result = results[0]  # 单帧结果
                    
                    # 修改检测结果中的bird标签为bottle
                    for box in result.boxes:
                        cls = int(box.cls)
                        if result.names[cls] == 'bird':
                            result.names[cls] = 'bottle'
                    
                    # 如果检测到物体
                    if len(result.boxes) > 0:
                        # 收集检测到的物体类型和位置信息
                        frame_detected_types = {}
                        frame_objects = []
                        
                        for box in result.boxes:
                            cls = int(box.cls)
                            type_name = result.names[cls]
                            conf = float(box.conf)
                            
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # 计算中心点坐标和尺寸
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            # 记录物体详细信息
                            object_info = {
                                "type": type_name,
                                "confidence": conf,
                                "position": {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "center_x": center_x,
                                    "center_y": center_y,
                                    "width": width,
                                    "height": height
                                }
                            }
                            frame_objects.append(object_info)
                            
                            # 更新全局统计
                            if type_name not in all_detected_types:
                                all_detected_types[type_name] = {
                                    "count": 1,
                                    "confidence": [conf],
                                    "total_frames": 1
                                }
                            else:
                                all_detected_types[type_name]["count"] += 1
                                all_detected_types[type_name]["confidence"].append(conf)
                                
                            # 更新当前帧统计
                            if type_name not in frame_detected_types:
                                frame_detected_types[type_name] = 1
                            else:
                                frame_detected_types[type_name] += 1
                        
                        # 在原图上绘制边界框和标签
                        annotated_frame = result.plot(
                            conf=True,  # 显示置信度
                            line_width=2,  # 边界框线条宽度
                            font_size=12  # 标签字体大小
                        )
                        
                        # 保存检测到物体的帧
                        if save_frames:
                            # 生成唯一文件名
                            now = datetime.datetime.now()
                            date_str = now.strftime("%Y%m%d_%H%M%S")
                            unique_id = str(uuid.uuid4()).replace("-", "")
                            frame_time = frame_index / fps if fps > 0 else 0
                            filename = f"{date_str}_{unique_id}_frame{frame_index}_time{frame_time:.2f}.jpg"
                            frame_path = os.path.join(self.history_path, filename)
                            
                            # 保存图片
                            cv2.imwrite(frame_path, annotated_frame)
                            
                            # 记录保存的帧信息
                            saved_frames.append({
                                "frame_index": frame_index,
                                "time": frame_time,
                                "path": frame_path,
                                "relative_path": f"/history/{Path(frame_path).name}",
                                "detected_types": frame_detected_types,
                                "objects": frame_objects,
                                "total_objects": len(frame_objects)
                            })
                
                frame_index += 1
                
                # 显示处理进度
                if frame_index % 100 == 0:
                    progress = frame_index / frame_count * 100 if frame_count > 0 else 0
                    logger.info(f"视频处理进度: {progress:.2f}%")
            
            # 释放资源
            cap.release()
            
            # 计算每种类型的平均置信度
            for type_name, data in all_detected_types.items():
                data["avg_confidence"] = sum(data["confidence"]) / len(data["confidence"])
                data["confidence"] = [round(c, 2) for c in data["confidence"]]
            
            # 生成结构化数据文件路径
            structured_data_path = None
            if saved_frames:
                # 生成唯一文件名
                now = datetime.datetime.now()
                date_str = now.strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4()).replace("-", "")[:8]
                structured_filename = f"structured_data_{date_str}_{unique_id}.json"
                structured_data_path = os.path.join(self.history_path, structured_filename)
                
                # 构建结构化数据
                structured_data = {
                    "metadata": {
                        "source_video": os.path.basename(video_path),
                        "processed_at": now.isoformat(),
                        "total_frames_processed": frame_index,
                        "total_frames_with_detections": len(saved_frames),
                        "detection_summary": all_detected_types
                    },
                    "frames": saved_frames
                }
                
                # 保存结构化数据到JSON文件
                with open(structured_data_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"已保存结构化数据到: {structured_data_path}")
            
            # 准备返回结果
            detection_result = {
                "success": True,
                "video_info": {
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration": duration,
                    "processed_frames": frame_index
                },
                "detected_objects": all_detected_types,
                "saved_frames": saved_frames,
                "total_saved_frames": len(saved_frames),
                "structured_data_path": structured_data_path
            }
            
            return detection_result
            
        except Exception as e:
            logger.error(f"处理视频时出错: {e}")
            return {"success": False, "error": str(e)}
    
    def save_to_database(self, db_connector, detection_result, task_id=None):
        """
        将检测结果保存到数据库
        
        Args:
            db_connector: 数据库连接器函数
            detection_result: 检测结果
            task_id: 关联的任务ID
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 检查检测结果是否有效
            if not detection_result.get("success", False):
                logger.error("无法保存无效的检测结果")
                return False
            
            # 处理单张图片的情况
            if "relative_path" in detection_result and detection_result["relative_path"]:
                # 获取检测到的所有类型
                detected_types = list(detection_result["detected_objects"].keys())
                if not detected_types:
                    logger.warning("没有检测到任何物体，不保存到数据库")
                    return False
                
                # 将所有类型合并为一个字符串
                types_str = ",".join(detected_types)
                
                # 获取图片路径
                image_path = detection_result["relative_path"]
                
                # 保存到数据库
                connection = db_connector()
                if connection:
                    with connection.cursor() as cursor:
                        # 插入历史记录
                        sql = """
                        INSERT INTO history (taskId, type, src, createdTime)
                        VALUES (%s, %s, %s, %s)
                        """
                        now = datetime.datetime.now()
                        cursor.execute(sql, (task_id, types_str, image_path, now))
                        connection.commit()
                        logger.info(f"历史记录已保存: {types_str}, {image_path}")
                    connection.close()
                    return True
                else:
                    logger.error("无法保存历史记录，数据库连接失败")
                    return False
            
            # 处理视频的情况
            elif "saved_frames" in detection_result and detection_result["saved_frames"]:
                # 获取保存的帧
                saved_frames = detection_result["saved_frames"]
                if not saved_frames:
                    logger.warning("没有保存任何视频帧，不保存到数据库")
                    return False
                
                # 保存到数据库
                connection = db_connector()
                if connection:
                    success_count = 0
                    
                    try:
                        # 为这次视频分析创建一个任务记录
                        task_sql = """
                        INSERT INTO analysis_tasks (source_video, start_time, status)
                        VALUES (%s, %s, %s)
                        """
                        task_id_value = task_id
                        if task_id_value is None:
                            # 从detection_result中获取视频路径
                            video_path = detection_result.get("video_info", {}).get("source_video", "unknown_video")
                            with connection.cursor() as cursor:
                                cursor.execute(task_sql, (os.path.basename(video_path), datetime.datetime.now(), "completed"))
                                task_id_value = cursor.lastrowid
                                connection.commit()
                        
                        # 批量插入帧记录
                        frame_sql = """
                        INSERT INTO video_frames (task_id, frame_index, time_seconds, image_path, detected_types, total_objects)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        
                        # 批量插入对象记录
                        object_sql = """
                        INSERT INTO detected_objects (frame_id, type, confidence, x1, y1, x2, y2, center_x, center_y, width, height)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        with connection.cursor() as cursor:
                            for frame in saved_frames:
                                # 插入帧记录
                                detected_types = list(frame["detected_types"].keys())
                                types_str = ",".join(detected_types)
                                
                                cursor.execute(frame_sql, (
                                    task_id_value,
                                    frame["frame_index"],
                                    frame["time"],
                                    frame["relative_path"],
                                    types_str,
                                    frame["total_objects"]
                                ))
                                
                                frame_id = cursor.lastrowid
                                
                                # 插入对象记录
                                for obj in frame["objects"]:
                                    cursor.execute(object_sql, (
                                        frame_id,
                                        obj["type"],
                                        obj["confidence"],
                                        obj["position"]["x1"],
                                        obj["position"]["y1"],
                                        obj["position"]["x2"],
                                        obj["position"]["y2"],
                                        obj["position"]["center_x"],
                                        obj["position"]["center_y"],
                                        obj["position"]["width"],
                                        obj["position"]["height"]
                                    ))
                                
                                success_count += 1
                            
                            connection.commit()
                            logger.info(f"成功保存 {success_count}/{len(saved_frames)} 个视频帧及其对象信息到数据库")
                        
                        # 如果有结构化数据文件，也记录下来
                        if detection_result.get("structured_data_path"):
                            structured_sql = """
                            INSERT INTO structured_data (task_id, file_path)
                            VALUES (%s, %s)
                            """
                            with connection.cursor() as cursor:
                                cursor.execute(structured_sql, (
                                    task_id_value,
                                    os.path.basename(detection_result["structured_data_path"])
                                ))
                                connection.commit()
                    except Exception as e:
                        logger.error(f"保存视频分析结果到数据库时出错: {e}")
                        connection.rollback()
                    finally:
                        connection.close()
                    
                    return success_count > 0
                else:
                    logger.error("无法保存历史记录，数据库连接失败")
                    return False
            
            else:
                logger.warning("检测结果中没有可保存的内容")
                return False
                
        except Exception as e:
            logger.error(f"保存检测结果到数据库时出错: {e}")
            return False