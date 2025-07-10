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
        <el-empty v-else-if="!isConnected && isConnecting" :description="getLoadingText()" v-loading="true" />
        
        <!-- 错误状态 -->
        <el-empty v-else-if="!isConnected" description="视频流连接失败">
          <template #extra>
            <el-button type="primary" @click="retryConnection" :loading="isConnecting">
              {{ isConnecting ? '连接中...' : '重试连接' }}
            </el-button>
          </template>
        </el-empty>
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
    
    // 验证数据完整性
    if (!data.image) {
      console.warn('收到的数据缺少图像信息')
      return
    }
    
    // 更新图像数据
    imageData.value = `data:image/jpeg;base64,${data.image}`
    
    // 更新流数据，使用默认值处理可能缺失的字段
    streamData.value = {
      fps: data.fps || 0,
      speed: data.speed || 0,
      weather: data.weather || '未知',
      source: data.source || '本地'
    }
    
    // 第一次收到数据时显示通知
    if (isConnecting.value) {
      isConnecting.value = false
      isConnected.value = true
    }
  } catch (error) {
    console.error('解析视频流数据失败:', error)
    // 不要因为单次解析错误就断开连接，继续等待有效数据
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
  imageData.value = null
  streamData.value = null
  
  // 设置连接超时
  const connectionTimeout = setTimeout(() => {
    if (ws.value && ws.value.readyState !== WebSocket.OPEN) {
      ElMessage.error('本地视频流连接超时')
      closeWebSocket()
      isConnecting.value = false
    }
  }, 10000) // 10秒超时
  
  try {
    ws.value = new WebSocket('ws://localhost:8081/ws/video')

    ws.value.onopen = () => {
      clearTimeout(connectionTimeout)
      isConnected.value = true
      isConnecting.value = false
      ElMessage.success('本地视频流连接成功')
      
      // 设置数据接收超时
      startDataTimeout()
    }

    ws.value.onmessage = (event) => {
      // 重置数据超时计时器
      resetDataTimeout()
      handleWebSocketMessage(event)
    }
    
    ws.value.onerror = (error) => {
      clearTimeout(connectionTimeout)
      handleWebSocketError(error)
    }
    
    ws.value.onclose = (event) => {
      clearTimeout(connectionTimeout)
      handleWebSocketClose('本地视频流', event)
    }
  } catch (error) {
    clearTimeout(connectionTimeout)
    ElMessage.error(`本地视频流连接失败: ${error.message}`)
    isConnecting.value = false
    console.error('WebSocket创建错误:', error)
  }
}

// 连接RTMP流
const connectRTMP = () => {
  closeWebSocket()
  isConnecting.value = true
  imageData.value = null
  streamData.value = null
  
  let wsUrl = 'ws://localhost:8081/ws/rtmp';
  
  // 如果提供了URL，则添加为查询参数
  if (rtmpUrl.value.trim()) {
    const encodedUrl = encodeURIComponent(rtmpUrl.value.trim())
    wsUrl = `${wsUrl}?rtmp_url=${encodedUrl}`;
  }
  
  // 设置连接超时
  const connectionTimeout = setTimeout(() => {
    if (ws.value && ws.value.readyState !== WebSocket.OPEN) {
      ElMessage.error('RTMP流连接超时，请检查流地址是否正确')
      closeWebSocket()
      isConnecting.value = false
    }
  }, 10000) // 10秒超时
  
  try {
    ws.value = new WebSocket(wsUrl)

    ws.value.onopen = () => {
      clearTimeout(connectionTimeout)
      isConnected.value = true
      isConnecting.value = false
      ElMessage.success('RTMP流连接成功')
      
      // 设置数据接收超时
      startDataTimeout()
    }

    ws.value.onmessage = (event) => {
      // 重置数据超时计时器
      resetDataTimeout()
      handleWebSocketMessage(event)
    }
    
    ws.value.onerror = (error) => {
      clearTimeout(connectionTimeout)
      handleWebSocketError(error)
    }
    
    ws.value.onclose = (event) => {
      clearTimeout(connectionTimeout)
      handleWebSocketClose('RTMP流', event)
    }
  } catch (error) {
    clearTimeout(connectionTimeout)
    ElMessage.error(`RTMP流连接失败: ${error.message}`)
    isConnecting.value = false
    console.error('WebSocket创建错误:', error)
  }
}

// 数据接收超时处理
let dataTimeoutId = null

const startDataTimeout = () => {
  dataTimeoutId = setTimeout(() => {
    if (isConnected.value && !streamData.value) {
      ElMessage.warning('RTMP流连接成功但未收到数据，请检查流地址是否有效')
      // 不断开连接，继续等待数据
    }
  }, 8000) // 8秒内没收到数据就提示
}

const resetDataTimeout = () => {
  if (dataTimeoutId) {
    clearTimeout(dataTimeoutId)
    dataTimeoutId = null
  }
}

// 切换流类型
const switchStreamType = (type) => {
  // 清除当前状态
  resetDataTimeout()
  imageData.value = null
  streamData.value = null
  isConnected.value = false
  
  // 根据类型连接不同的流
  if (type === 'local') {
    connectLocalVideo()
  } else if (type === 'rtmp') {
    connectRTMP()
  }
}

// 重试当前连接
const retryConnection = () => {
  if (streamType.value === 'local') {
    connectLocalVideo()
  } else if (streamType.value === 'rtmp') {
    connectRTMP()
  }
}

// 关闭WebSocket连接
const closeWebSocket = () => {
  // 清除所有超时计时器
  resetDataTimeout()
  
  if (ws.value) {
    // 移除所有事件监听器
    ws.value.onopen = null
    ws.value.onmessage = null
    ws.value.onerror = null
    ws.value.onclose = null
    
    // 关闭连接
    if (ws.value.readyState === WebSocket.OPEN || 
        ws.value.readyState === WebSocket.CONNECTING) {
      ws.value.close()
    }
    ws.value = null
  }
}

// 组件挂载时连接本地视频
onMounted(() => {
  connectLocalVideo()
})

// 组件卸载时清理资源
onUnmounted(() => {
  // 清除所有超时计时器
  resetDataTimeout()
  // 关闭WebSocket连接
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