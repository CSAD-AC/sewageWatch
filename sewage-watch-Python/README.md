# 污水监控视频流处理系统

## 功能特性

- **本地视频处理**: 支持本地视频文件和摄像头输入
- **RTMP流接收**: 支持接收RTMP实时视频流
- **YOLOv8目标检测**: 实时检测视频中的目标对象
- **WebSocket实时传输**: 通过WebSocket向前端实时传输处理后的视频帧

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务

```bash
python main.py
```

服务将在 `http://localhost:8081` 启动

## API端点

### 1. 本地视频流
- **WebSocket**: `ws://localhost:8081/ws/video`
- **功能**: 处理本地视频文件或摄像头输入

### 2. RTMP视频流
- **WebSocket**: `ws://localhost:8081/ws/rtmp?rtmp_url=<RTMP_URL>`
- **功能**: 接收并处理RTMP实时视频流
- **参数**: `rtmp_url` - RTMP流地址

### 3. API信息
- **HTTP GET**: `http://localhost:8081/`
- **功能**: 获取API端点信息

## RTMP使用示例

### 1. 连接RTMP流
```javascript
const rtmpUrl = 'rtmp://example.com/live/stream';
const encodedUrl = encodeURIComponent(rtmpUrl);
const ws = new WebSocket(`ws://localhost:8081/ws/rtmp?rtmp_url=${encodedUrl}`);
```

### 2. 常见RTMP流地址格式
- **标准RTMP**: `rtmp://server.com/live/streamkey`
- **带认证**: `rtmp://username:password@server.com/live/streamkey`
- **自定义端口**: `rtmp://server.com:1935/live/streamkey`

### 3. 移动端推流示例
如果你的移动端已经实现了RTMP推流，可以使用以下格式的URL：
- **OBS推流**: `rtmp://your-server.com/live/your-stream-key`
- **移动端推流**: 确保推流地址与接收地址匹配

## 数据格式

WebSocket返回的数据格式：
```json
{
  "image": "base64编码的JPEG图像",
  "fps": 30.0,
  "speed": 12.5,
  "weather": "晴朗",
  "source": "RTMP" // 或 "本地"
}
```

## 注意事项

1. **RTMP流稳定性**: RTMP流可能因网络问题中断，系统会自动尝试重连
2. **性能优化**: 使用队列缓冲机制，避免帧积压
3. **资源管理**: 自动释放断开连接的资源
4. **错误处理**: 完善的错误处理和日志记录

## 故障排除

### RTMP连接失败
1. 检查RTMP URL格式是否正确
2. 确认RTMP服务器是否可访问
3. 检查网络防火墙设置
4. 查看服务器日志获取详细错误信息

### 视频流卡顿
1. 检查网络带宽
2. 调整视频质量参数
3. 优化服务器性能

### 目标检测效果不佳
1. 调整YOLOv8的置信度阈值（conf参数）
2. 调整IOU阈值（iou参数）
3. 使用更适合的YOLOv8模型