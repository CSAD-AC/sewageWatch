<template>
  <div class="monitor-content" :class="{ 'loading': !isConnected }">
        <!-- 视频显示区域 -->
        <img v-if="imageData" :src="imageData" alt="视频流" class="video-stream" />
        
        <!-- 加载状态 -->
        <el-empty v-else-if="!isConnected" description="正在连接视频流..." v-loading="true" />
        
        <!-- 错误状态 -->
        <el-empty v-else description="视频流连接失败" />

        <!-- 数据展示 -->
        <div v-if="streamData" class="stream-info">
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
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'

const ws = ref(null)
const isConnected = ref(false)
const imageData = ref(null)
const streamData = ref(null)

// 连接WebSocket
const connectWebSocket = () => {
  ws.value = new WebSocket('ws://localhost:8081/ws/video')

  ws.value.onopen = () => {
    isConnected.value = true
    ElMessage.success('视频流连接成功')
  }

  ws.value.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      imageData.value = `data:image/jpeg;base64,${data.image}`
      streamData.value = {
        fps: data.fps,
        speed: data.speed,
        weather: data.weather
      }
    } catch (error) {
      console.error('解析视频流数据失败:', error)
    }
  }

  ws.value.onerror = (error) => {
    isConnected.value = false
    ElMessage.error('视频流连接错误')
    console.error('WebSocket错误:', error)
  }

  ws.value.onclose = () => {
    isConnected.value = false
    ElMessage.warning('视频流连接已关闭')
  }
}

// 关闭WebSocket连接
const closeWebSocket = () => {
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.close()
  }
}

// 组件挂载时连接WebSocket
onMounted(() => {
  connectWebSocket()
})

// 组件卸载时关闭WebSocket
onUnmounted(() => {
  closeWebSocket()
})
</script>

<style scoped>
.monitor-content {
  height: calc(100vh - 64px); /* 减去导航栏高度 */
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #000;
  overflow: hidden;
}

.monitor-content.loading {
  background-color: #1a1a1a;
}

.monitor-content :deep(.el-empty) {
  background-color: transparent;
  padding: 0;
}

.monitor-content :deep(.el-empty__description) {
  color: #fff;
}

.video-stream {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.stream-info {
  position: absolute;
  top: 16px;
  right: 16px;
  background-color: rgba(0, 0, 0, 0.75);
  padding: 12px 16px;
  border-radius: 4px;
  color: #fff;
  font-size: 14px;
  backdrop-filter: blur(4px);
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.3);
}

.monitor-content :deep(.el-loading-spinner .el-loading-text) {
  color: #fff;
}

.monitor-content :deep(.el-loading-spinner .path) {
  stroke: #fff;
}

.info-item {
  margin-bottom: 8px;
}

.info-item:last-child {
  margin-bottom: 0;
}

.label {
  margin-right: 8px;
  color: #909399;
}

.value {
  font-weight: bold;
}
</style>