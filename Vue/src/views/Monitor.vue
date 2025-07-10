<template>
  <div class="monitor-container">
    <!-- 控制面板 -->
    <div class="control-panel">
      <div class="stream-controls">
        <el-radio-group v-model="streamType" @change="switchStreamType">
          <el-radio-button label="local">本地视频</el-radio-button>
          <el-radio-button label="rtmp">RTMP流</el-radio-button>
        </el-radio-group>
        
        <div v-if="streamType === 'rtmp'" class="rtmp-input">
          <el-input
            v-model="rtmpUrl"
            placeholder="请输入RTMP流地址（可选，留空使用默认地址）"
            class="rtmp-url-input"
            @keyup.enter="connectRTMP"
          >
            <template #append>
              <el-button @click="connectRTMP" :disabled="isConnecting">
                {{ isConnecting ? '连接中...' : '连接' }}
              </el-button>
            </template>
          </el-input>
        </div>
      </div>
    </div>

    <!-- 视频显示区域 -->
    <div class="monitor-content" :class="{ 'loading': !isConnected }">
      <!-- 视频容器 -->
      <div class="video-container">
        <img v-if="imageData" :src="imageData" alt="视频流" class="video-stream" />
        
        <!-- 加载状态 -->
        <el-empty v-else-if="!isConnected" :description="getLoadingText()" v-loading="true" />
        
        <!-- 错误状态 -->
        <el-empty v-else description="视频流连接失败" />
      </div>

      <!-- 数据展示 -->
      <div v-if="streamData" class="stream-info">
        <div class="info-item">
          <span class="label">来源:</span>
          <span class="value">{{ getSourceText() }}</span>
        </div>
        <div class="info-item">
          <span class="label">FPS:</span>
          <span class="value">{{ streamData.fps }}</span>
        </div>
        <div class="info-item">
          <span class="label">速度:</span>
          <span class="value">{{ streamData.speed }} km/h</span>
        </div>
        <div class="info-item">
          <span class="label">天气:</span>
          <span class="value">{{ streamData.weather }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'

const ws = ref(null)
const isConnected = ref(false)
const isConnecting = ref(false)
const imageData = ref(null)
const streamData = ref(null)
const streamType = ref('local') // 'local' 或 'rtmp'
const rtmpUrl = ref('')

// 获取加载文本
const getLoadingText = () => {
  if (streamType.value === 'rtmp') {
    return '正在连接RTMP流...'
  }
  return '正在连接本地视频流...'
}

// 获取来源文本
const getSourceText = () => {
  if (!streamData.value) return ''
  
  if (streamType.value === 'local') {
    return '本地视频'
  } else {
    return rtmpUrl.value ? '自定义RTMP流' : '默认RTMP流'
  }
}

// 处理WebSocket消息
const handleWebSocketMessage = (event) => {
  try {
    const data = JSON.parse(event.data)
    imageData.value = `data:image/jpeg;base64,${data.image}`
    streamData.value = {
      fps: data.fps,
      speed: data.speed,
      weather: data.weather,
      source: data.source || '本地'
    }
  } catch (error) {
    console.error('解析视频流数据失败:', error)
  }
}

// 处理WebSocket错误
const handleWebSocketError = (error) => {
  isConnected.value = false
  isConnecting.value = false
  ElMessage.error(`${streamType.value === 'local' ? '本地视频' : 'RTMP'}流连接错误`)
  console.error('WebSocket错误:', error)
}

// 处理WebSocket关闭
const handleWebSocketClose = (sourceName, event) => {
  isConnected.value = false
  isConnecting.value = false
  
  if (event && event.code === 1008) {
    ElMessage.error(`${sourceName}连接失败: ${event.reason}`)
  } else if (streamType.value === 'local' && sourceName === '本地视频流' || 
            (streamType.value === 'rtmp' && sourceName === 'RTMP流')) {
    ElMessage.warning(`${sourceName}连接已关闭`)
  }
}

// 连接本地视频WebSocket
const connectLocalVideo = () => {
  closeWebSocket()
  isConnecting.value = true
  
  ws.value = new WebSocket('ws://localhost:8081/ws/video')

  ws.value.onopen = () => {
    isConnected.value = true
    isConnecting.value = false
    ElMessage.success('本地视频流连接成功')
  }

  ws.value.onmessage = handleWebSocketMessage
  ws.value.onerror = handleWebSocketError
  ws.value.onclose = (event) => handleWebSocketClose('本地视频流', event)
}

// 连接RTMP流
const connectRTMP = () => {
  closeWebSocket()
  isConnecting.value = true
  
  let wsUrl = 'ws://localhost:8081/ws/rtmp';
  
  // 如果提供了URL，则添加为查询参数
  if (rtmpUrl.value.trim()) {
    const encodedUrl = encodeURIComponent(rtmpUrl.value.trim())
    wsUrl = `${wsUrl}?rtmp_url=${encodedUrl}`;
  }
  
  ws.value = new WebSocket(wsUrl)

  ws.value.onopen = () => {
    isConnected.value = true
    isConnecting.value = false
    ElMessage.success('RTMP流连接成功')
  }

  ws.value.onmessage = handleWebSocketMessage
  ws.value.onerror = handleWebSocketError
  ws.value.onclose = (event) => handleWebSocketClose('RTMP流', event)
}

// 切换流类型
const switchStreamType = (type) => {
  imageData.value = null
  streamData.value = null
  
  if (type === 'local') {
    connectLocalVideo()
  } else if (type === 'rtmp') {
    connectRTMP()
  }
}

// 关闭WebSocket连接
const closeWebSocket = () => {
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.close()
  }
  ws.value = null
}

// 组件挂载时连接本地视频
onMounted(() => {
  connectLocalVideo()
})

// 组件卸载时关闭WebSocket
onUnmounted(() => {
  closeWebSocket()
})
</script>

<style scoped>
.monitor-container {
  display: flex;
  flex-direction: column;
  height: 100vh; /* 使用视口高度 */
  padding: 20px;
  box-sizing: border-box;
}

.control-panel {
  margin-bottom: 20px;
}

.stream-controls {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.rtmp-input {
  margin-top: 10px;
  max-width: 600px;
}

.monitor-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: #f5f7fa;
  border-radius: 4px;
  overflow: hidden;
  width: 100%;
  height: calc(100vh - 120px); /* 减去控制面板和padding的高度 */
}

.monitor-content.loading {
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background-color: #f5f7fa;
}

.video-stream {
  width: 1800px; /* 确保横向填满 */
  height: auto;
  max-height: 100%;
  object-fit: contain; /* 保持原始宽高比，不裁剪内容 */
}

.stream-info {
  background-color: rgba(0, 0, 0, 0.7); /* 恢复半透明黑色背景 */
  color: white;
  padding: 10px 20px;
  display: flex;
  justify-content: space-between;
}

.info-item {
  display: flex;
  align-items: center;
}

.label {
  font-weight: bold;
  margin-right: 5px;
}

.value {
  font-family: monospace;
}
</style>