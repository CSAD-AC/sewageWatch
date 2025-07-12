<template>
  <div class="detection-container">
    <h2>手动检测</h2>
    <el-card class="detection-card">
      <div class="upload-area">
        <el-upload
          class="upload-demo"
          drag
          action="#"
          :auto-upload="false"
          :on-change="handleFileChange"
          :before-remove="handleBeforeRemove"
          :file-list="fileList"
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            拖拽文件到此处或 <em>点击上传</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              支持上传无人机拍摄的图片(.jpg, .png, .jpeg)或视频文件(.mp4, .avi, .mov)
            </div>
          </template>
        </el-upload>
      </div>
      
      <div class="detection-actions">
        <el-button type="primary" :disabled="!hasFile || isDetecting" @click="startDetection">
          开始检测
        </el-button>
        <el-button :disabled="isDetecting" @click="resetDetection">重置</el-button>
      </div>
      
      <div v-if="isDetecting" class="detection-progress">
        <el-progress :percentage="detectionProgress" />
        <p>正在进行污水检测分析，请稍候...</p>
      </div>
      
      <div v-if="showResult && detectionResult" class="detection-result">
        <h3>检测结果</h3>
        
        <el-result
          :icon="detectionResult.success ? 'success' : 'error'"
          :title="detectionResult.success ? '检测完成' : '检测失败'"
          :sub-title="detectionResult.success ? '已成功完成污水检测分析' : detectionResult.error"
        >
          <template v-if="detectionResult.success" #extra>
            <div class="result-summary">
              <el-descriptions title="检测摘要" :column="2" border>
                <el-descriptions-item label="检测物体总数">
                  {{ detectionResult.total_detections || '未知' }}
                </el-descriptions-item>
                <el-descriptions-item label="检测时间">
                  {{ new Date().toLocaleString() }}
                </el-descriptions-item>
              </el-descriptions>
              
              <h4>检测到的物体类型</h4>
              <el-table :data="detectedObjectsArray" style="width: 100%">
                <el-table-column prop="type" label="类型" />
                <el-table-column prop="count" label="数量" />
                <el-table-column prop="avgConfidence" label="平均置信度">
                  <template #default="scope">
                    {{ (scope.row.avgConfidence * 100).toFixed(1) }}%
                  </template>
                </el-table-column>
              </el-table>
              
              <div v-if="detectionResult.result_path || (detectionResult.saved_frames && detectionResult.saved_frames.length > 0)" class="result-images">
                <h4>检测结果图片</h4>
                
                <!-- 单张图片结果 -->
                <div v-if="detectionResult.relative_path" class="result-image">
                  <el-image 
                    :src="getImageUrl(detectionResult.relative_path)" 
                    :preview-src-list="[getImageUrl(detectionResult.relative_path)]"
                    fit="contain"
                    style="max-height: 400px; max-width: 100%;"
                  />
                </div>
                
                <!-- 视频帧结果 -->
                <div v-if="detectionResult.saved_frames && detectionResult.saved_frames.length > 0" class="result-frames">
                  <el-carousel :interval="4000" type="card" height="400px">
                    <el-carousel-item v-for="(frame, index) in detectionResult.saved_frames" :key="index">
                      <el-image 
                        :src="getImageUrl(frame.relative_path)" 
                        fit="contain"
                        style="height: 100%; width: 100%;"
                      />
                      <div class="frame-info">
                        <p>帧: {{ frame.frame_index }} | 时间: {{ frame.time.toFixed(2) }}秒</p>
                        <p>检测到: {{ getFrameDetectedTypes(frame) }}</p>
                      </div>
                    </el-carousel-item>
                  </el-carousel>
                </div>
              </div>
              
              <el-button type="primary" @click="viewDetailedReport">查看详细报告</el-button>
              <el-button @click="downloadResult">下载检测结果</el-button>
            </div>
          </template>
        </el-result>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { ElMessage, ElLoading } from 'element-plus'
import axios from 'axios'

const fileList = ref([])
const hasFile = ref(false)
const isDetecting = ref(false)
const detectionProgress = ref(0)
const showResult = ref(false)
const detectionResult = ref(null)
const selectedFile = ref(null)

// 计算检测到的物体数组
const detectedObjectsArray = computed(() => {
  if (!detectionResult.value || !detectionResult.value.detected_objects) {
    return []
  }
  
  return Object.entries(detectionResult.value.detected_objects).map(([type, data]) => ({
    type,
    count: data.count,
    avgConfidence: data.avg_confidence
  }))
})

const handleFileChange = (file) => {
  // 清除之前的文件
  fileList.value = [file.raw]
  selectedFile.value = file.raw
  hasFile.value = true
  
  // 重置检测状态
  showResult.value = false
  detectionResult.value = null
}

const handleBeforeRemove = () => {
  // 重置所有状态
  resetDetection()
  return true
}

const startDetection = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择文件')
    return
  }
  
  isDetecting.value = true
  showResult.value = false
  detectionProgress.value = 0
  
  // 创建FormData对象
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  
  // 确定API端点
  const isImage = selectedFile.value.type.startsWith('image/')
  // 使用完整的API URL，确保能正确连接到后端
  const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8081'
  const apiUrl = `${baseUrl}/detect/${isImage ? 'image' : 'video'}`
  
  // 如果是视频，添加帧间隔参数
  if (!isImage) {
    formData.append('frame_interval', '30')  // 默认每30帧处理一次
  }
  
  // 创建一个可以取消的请求
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000); // 60秒超时
  
  let loadingInstance = null;
  let progressInterval = null;
  
  try {
    // 显示加载中
    loadingInstance = ElLoading.service({
      lock: true,
      text: '正在进行检测分析...',
      background: 'rgba(0, 0, 0, 0.7)'
    })
    
    // 模拟进度
    progressInterval = setInterval(() => {
      if (detectionProgress.value < 90) {
        detectionProgress.value += isImage ? 10 : 5
      }
    }, isImage ? 300 : 500)
    
    // 发送请求，添加超时和取消功能
    const response = await axios.post(apiUrl, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      signal: controller.signal,
      timeout: 60000 // 60秒超时
    })
    
    // 清除超时计时器
    clearTimeout(timeoutId);
    
    // 清除进度模拟
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
    
    detectionProgress.value = 100;
    
    // 处理响应
    detectionResult.value = response.data;
    showResult.value = true;
    
    if (response.data.success) {
      ElMessage.success('检测完成');
    } else {
      ElMessage.error(`检测失败: ${response.data.error || '未知错误'}`);
    }
    
  } catch (error) {
    // 清除进度模拟
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
    
    console.error('检测请求失败:', error);
    
    // 根据错误类型提供不同的错误信息
    let errorMessage = '检测请求失败';
    
    if (error.name === 'AbortError' || error.code === 'ECONNABORTED') {
      errorMessage = '请求超时，请检查网络连接或稍后重试';
    } else if (error.response) {
      // 服务器返回了错误状态码
      errorMessage = `服务器错误 (${error.response.status}): ${error.response.data?.error || error.message}`;
    } else if (error.request) {
      // 请求已发送但没有收到响应
      errorMessage = '无法连接到服务器，请检查后端服务是否正常运行';
    } else {
      // 请求设置时发生错误
      errorMessage = error.message || '未知错误';
    }
    
    detectionResult.value = {
      success: false,
      error: errorMessage
    };
    
    showResult.value = true;
    ElMessage.error(errorMessage);
    
  } finally {
    // 清除超时计时器（如果还存在）
    clearTimeout(timeoutId);
    
    // 关闭加载中
    if (loadingInstance) {
      loadingInstance.close();
    }
    
    isDetecting.value = false;
  }
}

const resetDetection = () => {
  fileList.value = []
  selectedFile.value = null
  hasFile.value = false
  isDetecting.value = false
  detectionProgress.value = 0
  showResult.value = false
  detectionResult.value = null
}

const viewDetailedReport = () => {
  ElMessage({
    message: '详细报告功能待完善',
    type: 'info'
  })
}

const downloadResult = () => {
  if (!detectionResult.value || !detectionResult.value.success) {
    ElMessage.warning('没有可下载的检测结果')
    return
  }
  
  // 创建一个包含检测结果的JSON文件
  const resultJson = JSON.stringify(detectionResult.value, null, 2)
  const blob = new Blob([resultJson], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  
  // 创建下载链接
  const a = document.createElement('a')
  a.href = url
  a.download = `detection_result_${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  
  ElMessage.success('检测结果已下载')
}

// 获取图片URL
const getImageUrl = (relativePath) => {
  // 假设后端服务器在同一域名下
  return relativePath
}

// 获取帧检测到的类型字符串
const getFrameDetectedTypes = (frame) => {
  if (!frame.detected_types) return '无'
  return Object.entries(frame.detected_types)
    .map(([type, count]) => `${type}(${count})`)
    .join(', ')
}
</script>

<style scoped>
.detection-container {
  padding: 20px;
}

.detection-card {
  margin-top: 20px;
}

.upload-area {
  margin: 20px 0;
}

.detection-actions {
  margin: 20px 0;
  display: flex;
  gap: 10px;
}

.detection-progress {
  margin: 30px 0;
}

.detection-result {
  margin-top: 30px;
  text-align: center;
}

.result-summary {
  width: 100%;
  text-align: left;
}

.result-images {
  margin: 20px 0;
}

.result-image {
  margin: 10px 0;
  display: flex;
  justify-content: center;
}

.result-frames {
  margin: 20px 0;
}

.frame-info {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 10px;
  text-align: center;
}
</style>