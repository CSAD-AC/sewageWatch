import cv2
import os
import uuid
import datetime
import logging
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionProcessor:
    """
    封装对图片和视频的模型处理逻辑
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
                        # 收集检测到的物体类型
                        frame_detected_types = {}
                        for box in result.boxes:
                            cls = int(box.cls)
                            type_name = result.names[cls]
                            conf = float(box.conf)
                            
                            # 更新全局统计
                            if type_name not in all_detected_types:
                                all_detected_types[type_name] = {
                                    "count": 1,
                                    "confidence": [conf]
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
                                "detected_types": frame_detected_types
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
                "total_saved_frames": len(saved_frames)
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
                    for frame in saved_frames:
                        # 获取检测到的所有类型
                        detected_types = list(frame["detected_types"].keys())
                        if not detected_types:
                            continue
                        
                        # 将所有类型合并为一个字符串
                        types_str = ",".join(detected_types)
                        
                        # 获取图片路径
                        image_path = frame["relative_path"]
                        
                        try:
                            with connection.cursor() as cursor:
                                # 插入历史记录
                                sql = """
                                INSERT INTO history (taskId, type, src, createdTime)
                                VALUES (%s, %s, %s, %s)
                                """
                                now = datetime.datetime.now()
                                cursor.execute(sql, (task_id, types_str, image_path, now))
                                connection.commit()
                                success_count += 1
                        except Exception as e:
                            logger.error(f"保存视频帧到数据库失败: {e}")
                    
                    connection.close()
                    logger.info(f"成功保存 {success_count}/{len(saved_frames)} 个视频帧到数据库")
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